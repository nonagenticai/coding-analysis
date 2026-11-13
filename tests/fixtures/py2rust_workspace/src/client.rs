pub async fn notify(client: &reqwest::Client, id: i64) {
    let base = std::env::var("SVC_A_URL").unwrap();
    client
        .post(format!("{}/api/orders/events/{}", base, id))
        .send()
        .await;
}

/// `c` is rebound to a non-client: not every binding is a client, so no call site.
pub async fn rebound() {
    let c = reqwest::Client::new();
    let c = std::collections::HashMap::<String, u8>::new();
    c.get("key");
}
