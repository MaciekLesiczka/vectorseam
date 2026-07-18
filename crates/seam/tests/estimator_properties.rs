use std::collections::BTreeMap;

use proptest::collection::{btree_set, vec};
use proptest::prelude::*;
use seam::math::{is_train_member, quantile_type7, select_ef, transfer_confidence};

proptest! {
    #[test]
    fn selected_ef_is_monotone_non_increasing_when_value_is_relaxed(
        quantiles in vec(0.0_f64..=1.0, 5),
        first_value in 1.0e-12_f64..=1.0,
        second_value in 1.0e-12_f64..=1.0,
    ) {
        let strict_value = first_value.max(second_value);
        let relaxed_value = first_value.min(second_value);
        let grid = [10, 20, 40, 80, 160]
            .into_iter()
            .zip(quantiles)
            .collect::<BTreeMap<_, _>>();

        let strict = select_ef(&grid, strict_value).unwrap();
        let relaxed = select_ef(&grid, relaxed_value).unwrap();

        prop_assert!(relaxed.recommended_ef <= strict.recommended_ef);
    }

    #[test]
    fn split_fraction_and_membership_are_stable_under_reordering(
        hashes in btree_set(any::<u64>(), 1..256),
        split_seed in any::<u64>(),
        train_fraction in 0.0001_f64..0.9999,
    ) {
        let forward = hashes
            .iter()
            .map(|hash| (*hash, is_train_member(*hash, split_seed, train_fraction).unwrap()))
            .collect::<BTreeMap<_, _>>();
        let reverse = hashes
            .iter()
            .rev()
            .map(|hash| (*hash, is_train_member(*hash, split_seed, train_fraction).unwrap()))
            .collect::<BTreeMap<_, _>>();

        prop_assert_eq!(&forward, &reverse);
        prop_assert_eq!(
            forward.values().filter(|member| **member).count(),
            reverse.values().filter(|member| **member).count()
        );
    }

    #[test]
    fn type7_quantile_is_bounded_by_input_extrema(
        values in vec(-1.0e6_f64..1.0e6, 1..128),
        q in 0.0_f64..=1.0,
    ) {
        let observed = quantile_type7(&values, q).unwrap();
        let minimum = values.iter().copied().min_by(f64::total_cmp).unwrap();
        let maximum = values.iter().copied().max_by(f64::total_cmp).unwrap();

        prop_assert!(observed >= minimum);
        prop_assert!(observed <= maximum);
    }

    #[test]
    fn confidence_is_monotone_in_successes_for_fixed_n(
        n in 1_usize..500,
        first_m in 0_usize..500,
        second_m in 0_usize..500,
        percentile in 0.01_f64..0.99,
    ) {
        let lower_m = first_m.min(second_m).min(n);
        let upper_m = first_m.max(second_m).min(n);
        let lower = transfer_confidence(n, lower_m, percentile).unwrap();
        let upper = transfer_confidence(n, upper_m, percentile).unwrap();

        prop_assert!(lower <= upper + 1e-12);
    }
}
