"""TypeScript/JavaScript provider routes, joinable with consumer call sites."""

from __future__ import annotations

import time

from processing.contract_analysis.call_sites import extract_file_call_sites
from processing.contract_analysis.parsers.ts_route_parser import (
    best_route,
    parse_ts_routes,
    route_matches,
)
from processing.contract_analysis.scanner import ContractScanner


def _keys(result):
    return [(r.method, r.path) for r in result.routes]


# fr backend shape: a method-first Router, route modules that take it as a parameter,
# a for-of method loop, a regex wildcard and a pathname dispatcher.
FR_SERVER = """\
import { Router } from './utils/router';
import { registerCodingRoutes } from './routes/coding/index';

const router = new Router();
router.add('GET', '/api/projects', handleListProjects);
router.add('GET', '/api/projects/:id', handleGetProject);
router.add(
  'PATCH',
  '/api/projects/:projectId/cards/:cardId/priority',
  handlePatch,
);
for (const method of ['GET', 'POST']) {
  router.add(method, '/api/assets/kag/reviews(.*)', proxy);
}
registerCodingRoutes(router);

Bun.serve({
  fetch(req) {
    const url = new URL(req.url);
    if (url.pathname === '/health') return ok();
    if (url.pathname.startsWith('/api/ws/')) return upgrade();
    return router.match(req.method, url.pathname);
  },
});
"""

FR_CODING = """\
export function registerCodingRoutes(router: Router): void {
  router.add('GET', '/api/coding/containers', handleList);
  router.add('POST', `/api/coding/dependencies/${':container'}/scan`, scan);
}
"""


def test_fr_method_first_router():
    result = parse_ts_routes({"src/server.ts": FR_SERVER, "src/routes/coding/index.ts": FR_CODING})
    assert sorted(_keys(result)) == sorted(
        [
            ("POST", "/api/coding/dependencies/{param}/scan"),
            ("GET", "/api/coding/containers"),
            ("GET", "/api/projects"),
            ("GET", "/api/projects/{param}"),
            ("PATCH", "/api/projects/{param}/cards/{param}/priority"),
            ("GET", "/api/assets/kag/reviews"),
            ("POST", "/api/assets/kag/reviews"),
            ("ALL", "/health"),
            ("ALL", "/api/ws"),
        ]
    )
    wild = {r.path for r in result.routes if r.wildcard}
    assert wild == {"/api/assets/kag/reviews", "/api/ws"}
    assert result.unresolved == []


def test_dynamic_method_and_path_are_unresolved():
    src = """\
const router = new Router();
router.add(verb, '/api/a', h);
router.add('GET', buildPath(), h);
"""
    result = parse_ts_routes({"s.ts": src})
    assert result.routes == []
    assert [(u.line, u.reason) for u in result.unresolved] == [
        (2, "dynamic route method"),
        (3, "dynamic route path"),
    ]


EXPRESS_APP = """\
import express from 'express';
import usersRouter from './routes/users';
import { adminRouter } from './routes/admin';
const v1 = require('./routes/v1');

const app = express();
const health = express.Router();
health.get('/live', (req, res) => res.send('ok'));

app.use('/api/users', usersRouter);
app.use('/api/admin', adminRouter);
app.use('/api/v1', v1);
app.use('/health', health);
app.use(express.json());
app.get('/', home);
app.route('/api/items').get(list).post(create);
"""

EXPRESS_USERS = """\
import { Router } from 'express';
const router = Router();
router.get('/', listUsers);
router.get('/:id', getUser);
router.delete('/:id', deleteUser);
export default router;
"""

EXPRESS_ADMIN = """\
import { Router } from 'express';
export const adminRouter = Router();
const audit = Router();
audit.get('/log', getLog);
adminRouter.use('/audit', audit);
adminRouter.post('/reindex', reindex);
"""

EXPRESS_V1 = """\
const express = require('express');
const router = express.Router();
router.all('/echo', echo);
module.exports = router;
"""

EXPRESS_ORPHAN = """\
import { Router } from 'express';
const router = Router();
router.get('/where', h);
export default router;
"""


def test_express_mounts_resolve_through_imports():
    result = parse_ts_routes(
        {
            "src/app.ts": EXPRESS_APP,
            "src/routes/users.ts": EXPRESS_USERS,
            "src/routes/admin.ts": EXPRESS_ADMIN,
            "src/routes/v1.js": EXPRESS_V1,
            "src/routes/orphan.ts": EXPRESS_ORPHAN,
        }
    )
    assert sorted(_keys(result)) == sorted(
        [
            ("GET", "/health/live"),
            ("GET", "/"),
            ("GET", "/api/items"),
            ("POST", "/api/items"),
            ("GET", "/api/admin/audit/log"),
            ("POST", "/api/admin/reindex"),
            ("GET", "/api/users"),
            ("GET", "/api/users/{param}"),
            ("DELETE", "/api/users/{param}"),
            ("ALL", "/api/v1/echo"),
        ]
    )
    assert [(u.file, u.reason) for u in result.unresolved] == [
        ("src/routes/orphan.ts", "exported router whose mount point is not in the scanned files")
    ]


def test_dynamic_mount_prefix_and_missing_target_are_unresolved():
    app = """\
import express from 'express';
import sub from './sub';
import ext from 'some-package';
const app = express();
app.use(process.env.PREFIX, sub);
app.use('/ext', ext);
"""
    sub = "import { Router } from 'express';\nconst r = Router();\nr.get('/x', h);\nexport default r;\n"
    result = parse_ts_routes({"app.ts": app, "sub.ts": sub})
    assert result.routes == []
    reasons = sorted(u.reason for u in result.unresolved)
    assert reasons == [
        "mount target is not a router in the scanned files",
        "mounted under a dynamic prefix",
    ]


def test_fastify_route_object_and_hono_on():
    src = """\
fastify.route({ method: ['GET', 'HEAD'], url: '/api/things/:id', handler });
app.on('PUT', '/api/things/:id', h);
"""
    assert _keys(parse_ts_routes({"f.ts": src})) == [
        ("GET", "/api/things/{param}"),
        ("HEAD", "/api/things/{param}"),
        ("PUT", "/api/things/{param}"),
    ]


def test_set_add_and_event_on_are_not_routes():
    src = "seen.add(key);\nsocket.on('message', h);\nemitter.on('GET', h);\n"
    result = parse_ts_routes({"n.ts": src})
    assert result.routes == [] and result.unresolved == []


def test_consumer_call_sites_join_ts_provider_routes():
    provider = parse_ts_routes(
        {"src/server.ts": FR_SERVER, "src/routes/coding/index.ts": FR_CODING}
    )
    consumer = """\
export const p = (id: string) => authFetch(`/api/projects/${id}`);
export const l = () => authFetch('/api/projects/');
export const r = () => authFetch('/api/assets/kag/reviews/er-1/decide', { method: 'POST' });
export const h = () => fetch('/health');
export const m = () => authFetch('/api/projects', { method: 'DELETE' });
"""
    sites = extract_file_call_sites("api.ts", consumer).call_sites
    joined = [
        (s.method, s.target, any(route_matches(s.method, s.target, r) for r in provider.routes))
        for s in sites
    ]
    assert joined == [
        ("GET", "/api/projects/{param}", True),
        ("GET", "/api/projects", True),
        ("POST", "/api/assets/kag/reviews/er-1/decide", True),
        ("GET", "/health", True),
        ("DELETE", "/api/projects", False),
    ]


def test_scan_contracts_accepts_typescript_route_sources():
    client = """\
class C:
    async def go(self):
        return await self._request("GET", "/api/projects")
    async def gone(self):
        return await self._request("GET", "/api/nowhere")
"""
    result = ContractScanner().scan_sources(
        name="py-client -> ts-service",
        client_source=client,
        schema_sources=[""],
        route_sources=[],
        ts_route_sources={"src/server.ts": FR_SERVER},
        check_syntax_enabled=False,
    )
    assert result.stats["service_routes"] == 7
    messages = [v.message for v in result.violations if v.check == "route_path"]
    assert len(messages) == 1 and "/api/nowhere" in messages[0]


def test_best_route_prefers_the_most_specific_hit():
    routes = parse_ts_routes(
        {
            "s.ts": (
                "router.add('GET', '/api/assets/architecture/elements/:id', a);\n"
                "router.add('GET', '/api/assets/:kind/:id/streams', b);\n"
                "router.add('GET', '/api/assets/streams/:id', c);\n"
                "router.add('GET', '/api/assets/streams/by-slug', d);\n"
            )
        }
    ).routes
    hit = best_route("GET", "/api/assets/{param}/{param}/streams", routes)
    assert hit is not None and hit.route.line == 2 and not hit.ambiguous and not hit.weak
    hit = best_route("GET", "/api/assets/streams/by-slug", routes)
    assert hit is not None and hit.route.line == 4 and not hit.ambiguous and not hit.weak
    hit = best_route("GET", "/api/assets/streams/s-1", routes)
    assert hit is not None and hit.route.line == 3 and not hit.ambiguous and not hit.weak
    assert [r.line for r in hit.candidates] == [3]
    assert best_route("POST", "/api/assets/streams/s-1", routes) is None


def test_best_route_tie_is_ambiguous_and_lists_every_candidate():
    routes = parse_ts_routes(
        {
            "s.ts": (
                "router.add('GET', '/api/assets/_assignments', a);\n"
                "router.add('GET', '/api/assets/_tags', b);\n"
                "router.add('GET', '/api/other/{x}', c);\n"
            )
        }
    ).routes
    hit = best_route("GET", "/api/assets/{param}", routes)
    assert hit is not None
    assert hit.ambiguous and hit.weak
    assert [r.line for r in hit.candidates] == [1, 2]
    assert hit.route is hit.candidates[0] and hit.score == 6  # "", api, assets equal


def test_best_route_consumer_param_on_route_literal_is_weak_even_untied():
    routes = parse_ts_routes({"s.ts": "router.add('GET', '/api/assets/_assignments', a);\n"}).routes
    hit = best_route("GET", "/api/assets/{param}", routes)
    assert hit is not None and hit.weak and not hit.ambiguous
    assert [r.line for r in hit.candidates] == [1]
    # A route {param} against a consumer literal is a normal (not weak) hit.
    routes = parse_ts_routes({"s.ts": "router.add('GET', '/api/assets/:id', a);\n"}).routes
    hit = best_route("GET", "/api/assets/_assignments", routes)
    assert hit is not None and not hit.weak and not hit.ambiguous


def test_brace_run_route_path_normalises_in_linear_time():
    source = "router.get('/" + "{" * (256 * 1024) + "', h);\n"
    start = time.perf_counter()
    parse_ts_routes({"a.ts": source})
    elapsed = time.perf_counter() - start
    assert elapsed < 6.0, f"{elapsed:.2f}s"
