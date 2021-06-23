/// Assumes the vector is sorted.
pub(crate) fn remove_doubles<T: Eq + Copy>(v: &mut Vec<T>) {
    let mut ii = 0;
    let mut jj = 0;
    while jj + 1 < v.len() {
        if v[jj] == v[jj + 1] {
            jj += 2;
        } else {
            v[ii] = v[jj];
            ii += 1;
            jj += 1;
        }
    }
    if jj < v.len() {
        v[ii] = v[jj];
        ii += 1;
        jj += 1;
    }
    while jj > ii {
        v.pop();
        jj -= 1;
    }
}

#[cfg(test)]
mod sc_tests {
    use super::*;

    #[test]
    fn test_remove_dups() {
        let mut v = vec![0, 0, 1, 2, 3, 3];
        remove_doubles(&mut v);
        assert_eq!(v, vec![1, 2])
    }

    #[test]
    fn test_remove_dups_again() {
        let mut v = vec![0, 0, 1, 2, 2, 3];
        remove_doubles(&mut v);
        assert_eq!(v, vec![1, 3])
    }
}
