/// variable key
#[derive(Hash, Eq, PartialEq, Ord, PartialOrd, Copy, Clone, Debug)]
pub struct Vkey(pub usize);
#[derive(Hash, Eq, PartialEq, Ord, PartialOrd, Copy, Clone, Debug)]
pub struct ExternVkey(pub usize);
/// factor key
pub struct Fkey(pub usize);
