use crate::db;
use crate::models::Item;
use axum::{routing::post, Router};

#[derive(Debug, serde::Serialize)]
pub struct ItemOut {
    pub id: i64,
    name: String,
}

pub async fn get_item(id: i64) -> Item {
    db::load(id).await
}

pub async fn delete_item(id: i64) {}

pub fn v1_router() -> Router {
    Router::new().route("/jobs/:job_id", post(create_job))
}

pub async fn create_job() {}
