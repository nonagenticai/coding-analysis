pub mod loops;
pub mod users;

use axum::{routing::get, Router};

pub async fn health() -> &'static str {
    "ok"
}

pub fn api_router() -> Router {
    Router::new()
        .route("/items/{id}", get(users::get_item).delete(users::delete_item))
        .nest("/v1", users::v1_router())
}
