use axum::{routing::get, Router};

pub async fn ping() {}

/// a() nests b(), which merges a() again: a router cycle.
pub fn a() -> Router {
    Router::new().route("/x", get(ping)).nest("/b", b())
}

pub fn b() -> Router {
    Router::new().merge(a())
}

/// `app` is rebound from itself: the let is followed once, not forever.
pub fn layered(app: Router) -> Router {
    let app = app.route("/layered", get(ping));
    app.route("/layered/more", get(ping))
}
