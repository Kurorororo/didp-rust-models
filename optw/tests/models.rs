use optw::{Instance, RoundedInstance, dypdl_model, rpid_model::Optw};
use rpid::{prelude::*, solvers};
use std::{rc::Rc, sync::Arc};

// Enumerate all partial tours without reachability pruning or heuristic bounds.
fn exhaustive(instance: &RoundedInstance, current: usize, time: i32, visited: u64) -> Option<i32> {
    if time > instance.closing[0] {
        return None;
    }
    let mut best = (time + instance.distances[current][0] <= instance.closing[0]).then_some(0);
    for next in 1..instance.vertices.len() {
        if visited & (1 << next) != 0 {
            continue;
        }
        let arrival = (time + instance.distances[current][next]).max(instance.opening[next]);
        if arrival > instance.closing[next] {
            continue;
        }
        if let Some(suffix) = exhaustive(instance, next, arrival, visited | (1 << next)) {
            let profit = instance.profits[next] + suffix;
            best = Some(best.map_or(profit, |value| value.max(profit)));
        }
    }
    best
}

fn check(instance: RoundedInstance) {
    let expected = exhaustive(&instance, 0, instance.opening[0].max(0), 1);
    {
        let dp = Optw::new(instance.clone(), 1e-6);
        if let Some(expected) = expected {
            assert!(dp.get_dual_bound(&dp.get_target()).unwrap() >= expected);
        }
        for threads in [1, 2] {
            let parameters = SearchParameters {
                quiet: true,
                ..Default::default()
            };
            let mut solver = if threads == 1 {
                solvers::create_cabs(dp.clone(), parameters, CabsParameters::default())
            } else {
                solvers::create_parallel_cabs(
                    dp.clone(),
                    parameters,
                    CabsParameters::default(),
                    threads,
                )
            };
            let solution = solver.search();
            assert_eq!(solution.cost, expected, "RPID: {instance:?}");
            if let Some(profit) = solution.cost {
                assert!(solution.is_optimal);
                let tour = solution
                    .transitions
                    .into_iter()
                    .filter(|&i| i < instance.vertices.len())
                    .collect::<Vec<_>>();
                assert!(instance.validate(&tour, profit));
            } else {
                assert!(solution.is_infeasible);
            }
        }
        let mut solver = solvers::create_astar(
            dp,
            SearchParameters {
                quiet: true,
                ..Default::default()
            },
        );
        assert_eq!(solver.search().cost, expected);
    }
    let model = dypdl_model::create_model(&instance, 1e-6);
    for threads in 0..=2 {
        let parameters = dypdl_heuristic_search::Parameters {
            quiet: true,
            ..Default::default()
        };
        let mut solver = if threads == 0 {
            dypdl_heuristic_search::create_caasdy::<i32>(
                Rc::new(model.clone()),
                parameters,
                dypdl_heuristic_search::FEvaluatorType::Plus,
            )
        } else {
            let parameters = dypdl_heuristic_search::CabsParameters {
                beam_search_parameters: dypdl_heuristic_search::BeamSearchParameters {
                    parameters,
                    ..Default::default()
                },
                ..Default::default()
            };
            if threads == 1 {
                dypdl_heuristic_search::create_dual_bound_cabs(
                    Rc::new(model.clone()),
                    parameters,
                    dypdl_heuristic_search::FEvaluatorType::Plus,
                )
            } else {
                dypdl_heuristic_search::create_dual_bound_cahdbs2(
                    Arc::new(model.clone()),
                    parameters,
                    dypdl_heuristic_search::FEvaluatorType::Plus,
                    threads,
                )
            }
        };
        let solution = solver.search().unwrap();
        assert_eq!(
            solution.cost, expected,
            "DyPDL with {threads} threads: {instance:?}"
        );
        if let Some(profit) = solution.cost {
            assert!(solution.is_optimal);
            let tour = solution
                .transitions
                .iter()
                .map(|t| t.get_full_name().parse::<usize>().unwrap())
                .filter(|&i| i < instance.vertices.len())
                .collect::<Vec<_>>();
            assert!(instance.validate(&tour, profit));
        } else {
            assert!(solution.is_infeasible);
        }
    }
}

#[test]
fn optw_matches_exhaustive_partial_tours() {
    // Includes zero travel times, negative rewards, waiting, service times,
    // depot opening times, and asymmetric nonmetric distances.
    let mut seed = 42_u64;
    let mut draw = |limit: i32| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 32) % limit as u64) as i32
    };
    for case in 0..64 {
        let n = 1 + case % 5;
        let service = (0..n).map(|_| draw(3)).collect::<Vec<_>>();
        let distances = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| service[i] + if i == j { 0 } else { draw(9) })
                    .collect()
            })
            .collect();
        let mut opening = (0..n).map(|_| draw(15)).collect::<Vec<_>>();
        opening[0] = draw(5);
        let closing = opening.iter().map(|&a| a + draw(16)).collect();
        let mut profits = (0..n).map(|_| draw(20) - 4).collect::<Vec<_>>();
        profits[0] = 0;
        check(RoundedInstance {
            vertices: (0..n).collect(),
            distances,
            profits,
            opening,
            closing,
        });
    }
}

#[test]
fn waiting_can_make_return_impossible_and_stopping_is_optional() {
    check(RoundedInstance {
        vertices: vec![0, 1, 2],
        distances: vec![vec![0, 1, 1], vec![1, 0, 1], vec![1, 1, 0]],
        profits: vec![0, 10, -5],
        opening: vec![0, 10, 0],
        closing: vec![10, 20, 20],
    });
    check(RoundedInstance {
        vertices: vec![0, 1, 2],
        distances: vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]],
        profits: vec![0, 10, -5],
        opening: vec![0; 3],
        closing: vec![0; 3],
    });
}

#[test]
fn rounded_nonmetric_paths_can_require_a_negative_profit_visit() {
    check(RoundedInstance {
        vertices: vec![0, 1, 2],
        distances: vec![vec![0, 1, 10], vec![10, 0, 1], vec![1, 10, 0]],
        profits: vec![0, -1, 10],
        opening: vec![0; 3],
        closing: vec![3; 3],
    });
}

#[test]
fn rounding_and_validation_match_the_reference_model() {
    let rounded = RoundedInstance::new(
        Instance {
            vertices: vec![0, 1],
            coordinates: vec![(0.0, 0.0), (0.16, 0.0)],
            service_time: vec![0.16, 0.0],
            profits: vec![0.0, 1.0],
            opening: vec![1.0, 0.0],
            closing: vec![2.0, 2.0],
        },
        1,
    );
    assert_eq!(rounded.distances[0][1], 2);
    assert!(rounded.validate(&[1], 1));
    assert!(!rounded.validate(&[0], 0));
    assert!(!rounded.validate(&[2], 0));
    assert!(!rounded.validate(&[1, 1], 2));
}
