use std::ops::Index;

use crate::core::key::Vkey;
use hashbrown::HashMap;
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Clone)]
pub struct VariableOrdering {
    keymap: HashMap<Vkey, usize>,
    keylist: Vec<Vkey>,
}
impl VariableOrdering {
    pub fn key(&self, index: usize) -> Option<Vkey> {
        self.keylist.get(index).copied()
    }
    pub fn len(&self) -> usize {
        self.keylist.len()
    }
    pub fn is_empty(&self) -> bool {
        self.keylist.is_empty()
    }
    pub fn keys(&self) -> &[Vkey] {
        &self.keylist
    }
    pub fn search_key(&self, key: Vkey) -> Option<usize> {
        self.keymap.get(&key).copied()
    }
}
impl Default for VariableOrdering {
    fn default() -> Self {
        VariableOrdering {
            keymap: HashMap::<Vkey, usize>::new(),
            keylist: Vec::<Vkey>::new(),
        }
    }
}
impl VariableOrdering {
    pub fn new(keylist: &[Vkey]) -> Self {
        let mut keymap: HashMap<Vkey, usize> = HashMap::new();
        for (i, k) in keylist.iter().enumerate() {
            if keymap.insert(*k, i).is_some() {
                panic!("keymap already has a key {:?}", k)
            }
        }
        VariableOrdering {
            keymap,
            keylist: keylist.to_owned(),
        }
    }
}
impl Index<usize> for VariableOrdering {
    type Output = Vkey;

    fn index(&self, index: usize) -> &Self::Output {
        &self.keylist[index]
    }
}
#[cfg(test)]
mod tests {
    use crate::core::key::Vkey;

    use super::VariableOrdering;

    #[test]
    fn key() {
        let keys = vec![Vkey(0), Vkey(1)];
        let var_ord = VariableOrdering::new(&keys);
        assert_eq!(var_ord.key(0).unwrap(), Vkey(0));
        assert_eq!(var_ord.key(1).unwrap(), Vkey(1));
        assert!(var_ord.key(2).is_none());
    }
    #[test]
    fn len() {
        let keys = vec![Vkey(0), Vkey(1)];
        let var_ord = VariableOrdering::new(&keys);
        assert_eq!(var_ord.len(), 2);
    }
    #[test]
    fn search_key() {
        let keys = vec![Vkey(0), Vkey(1)];
        let var_ord = VariableOrdering::new(&keys);
        assert_eq!(var_ord.search_key(Vkey(0)).unwrap(), 0);
        assert_eq!(var_ord.search_key(Vkey(1)).unwrap(), 1);
    }
    #[test]
    #[should_panic]
    fn new_with_duplicate_keys() {
        let keys = vec![Vkey(0), Vkey(1), Vkey(1)];
        let _var_ord = VariableOrdering::new(&keys);
    }
}
