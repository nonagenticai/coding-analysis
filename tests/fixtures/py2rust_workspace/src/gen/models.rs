pub struct Item {
    pub id: i64,
    pub name: String,
}

impl Item {
    pub fn label(&self) -> &str {
        &self.name
    }
}

#[macro_export]
macro_rules! item_id {
    ($item:expr) => {
        $item.id
    };
}
