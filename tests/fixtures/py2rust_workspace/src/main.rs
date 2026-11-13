mod client;
mod db;
mod routes;
#[path = "gen/models.rs"]
mod models;

use axum::Router;

macro_rules! make_handler {
    ($name:ident) => {
        pub async fn $name() -> &'static str { "ok" }
    };
}
make_handler!(generated_ping);

pub fn app() -> Router {
    Router::new()
        .route("/health", axum::routing::get(routes::health))
        .nest("/api", routes::api_router())
        .nest("/a", Router::new().nest("/b", leaf()))
        .nest("/loop", routes::loops::a())
        .nest_service("/assets", tower_http::services::ServeDir::new("assets"))
}

fn leaf() -> Router {
    Router::new().route("/c", axum::routing::get(routes::health))
}

#[tokio::main]
async fn main() {
    tokio::spawn(async { db::cleanup_loop().await });
    let app = app();
    axum::serve(listener, app).await.unwrap();
}
