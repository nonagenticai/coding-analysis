use crate::models::Item;

pub async fn load(id: i64) -> Item {
    sqlx::query_as!(Item, "SELECT id, name FROM items WHERE id = $1", id);
    todo!()
}

pub async fn cleanup_loop() {
    sqlx::query!("DELETE FROM jobs WHERE done");
}

pub async fn purge(pool: &sqlx::PgPool) {
    pool.execute("DELETE FROM sessions WHERE expired").await;
}
