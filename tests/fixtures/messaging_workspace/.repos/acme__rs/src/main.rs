use redis::Commands;

const ALERTS: &str = "alerts";

fn send(con: &mut redis::Connection) -> redis::RedisResult<()> {
    con.publish(ALERTS, "x")?;
    con.publish(format!("user.{}.events", 1), "y")?;
    redis::cmd("XADD").arg("payments").arg("*").arg("k").arg("v").query::<()>(con)?;
    con.publish(std::env::var("RS_CHAN").unwrap(), "z")?;
    Ok(())
}

fn listen(pubsub: &mut redis::PubSub) {
    pubsub.subscribe("orders.created").unwrap();
    pubsub.psubscribe("alert*").unwrap();
}

fn main() {}
