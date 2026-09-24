use crate::RoundedInstance;
use fixedbitset::FixedBitSet;
use rpid::{algorithms, prelude::*};
use std::cmp::Ordering;

#[derive(Clone)]
pub struct Optw {
    instance: RoundedInstance,
    shortest: Vec<Vec<i32>>,
    min_from: Vec<i32>,
    min_to: Vec<i32>,
    sorted_from: Vec<(usize, i32, i32)>,
    sorted_to: Vec<(usize, i32, i32)>,
    epsilon: f64,
}

#[derive(Clone)]
pub struct OptwState {
    reachable: FixedBitSet,
    current: usize,
    time: i32,
}

impl Optw {
    pub fn new(instance: RoundedInstance, epsilon: f64) -> Self {
        assert!(
            epsilon.is_finite() && epsilon >= 0.0,
            "epsilon must be finite and nonnegative"
        );
        let shortest = crate::compute_pairwise_shortest_path_costs(&instance.distances);
        let min_from = algorithms::take_row_wise_min_without_diagonal(&instance.distances)
            .map(|d| d.unwrap_or(0))
            .collect::<Vec<_>>();
        let min_to = algorithms::take_column_wise_min_without_diagonal(&instance.distances)
            .map(|d| d.unwrap_or(0))
            .collect::<Vec<_>>();
        // Skipping a customer is allowed, so negative profits cannot tighten an upper bound.
        let rewards = instance
            .profits
            .iter()
            .map(|&p| p.max(0))
            .collect::<Vec<_>>();
        let sorted_from = algorithms::sort_knapsack_items_by_efficiency(&min_from, &rewards);
        let sorted_to = algorithms::sort_knapsack_items_by_efficiency(&min_to, &rewards);
        Self {
            instance,
            shortest,
            min_from,
            min_to,
            sorted_from,
            sorted_to,
            epsilon,
        }
    }

    fn can_reach(&self, current: usize, time: i32, next: usize) -> bool {
        let earliest = (time + self.shortest[current][next]).max(self.instance.opening[next]);
        earliest <= self.instance.closing[next]
            && earliest + self.shortest[next][0] <= self.instance.closing[0]
    }

    fn knapsack_bound(&self, state: &OptwState, capacity: i32, items: &[(usize, i32, i32)]) -> i32 {
        let capacity = capacity.max(0);
        let items = items
            .iter()
            .filter(|&&(i, _, value)| state.reachable.contains(i) && value > 0);
        let bound = algorithms::compute_fractional_knapsack_profit(
            capacity,
            items.map(|&(_, weight, value)| (weight, value)),
        );
        (bound + self.epsilon).floor() as i32
    }
}

impl Dp for Optw {
    type State = OptwState;
    type CostType = i32;
    type Label = usize;

    fn get_target(&self) -> Self::State {
        let time = self.instance.opening[0].max(0);
        let mut reachable = FixedBitSet::with_capacity(self.instance.vertices.len());
        for next in 1..self.instance.vertices.len() {
            if self.can_reach(0, time, next) {
                reachable.insert(next);
            }
        }
        OptwState {
            reachable,
            current: 0,
            time,
        }
    }

    fn get_successors(
        &self,
        state: &Self::State,
    ) -> impl IntoIterator<Item = (Self::State, i32, usize)> {
        let goal = self.instance.vertices.len();
        if state.current == goal || state.time > self.instance.closing[0] {
            return Vec::new();
        }
        let mut successors = Vec::new();
        for next in state.reachable.ones() {
            let time = (state.time + self.instance.distances[state.current][next])
                .max(self.instance.opening[next]);
            if time > self.instance.closing[next]
                || time + self.shortest[next][0] > self.instance.closing[0]
            {
                continue;
            }
            let mut reachable = state.reachable.clone();
            reachable.remove(next);
            for candidate in state.reachable.ones() {
                if !self.can_reach(next, time, candidate) {
                    reachable.remove(candidate);
                }
            }
            successors.push((
                OptwState {
                    reachable,
                    current: next,
                    time,
                },
                self.instance.profits[next],
                next,
            ));
        }
        // Returning is optional and collects no profit; it does not require visiting every customer.
        let time = state.time + self.instance.distances[state.current][0];
        if time <= self.instance.closing[0] {
            successors.push((
                OptwState {
                    reachable: FixedBitSet::with_capacity(goal),
                    current: goal,
                    time,
                },
                0,
                goal,
            ));
        }
        successors
    }

    fn get_base_cost(&self, state: &Self::State) -> Option<i32> {
        (state.current == self.instance.vertices.len()).then_some(0)
    }

    fn get_optimization_mode(&self) -> OptimizationMode {
        OptimizationMode::Maximization
    }
}

impl Dominance for Optw {
    type State = OptwState;
    type Key = usize;

    fn get_key(&self, state: &Self::State) -> usize {
        state.current
    }

    fn compare(&self, a: &Self::State, b: &Self::State) -> Option<Ordering> {
        if a.time == b.time && a.reachable == b.reachable {
            Some(Ordering::Equal)
        } else if a.time <= b.time && b.reachable.is_subset(&a.reachable) {
            Some(Ordering::Greater)
        } else if b.time <= a.time && a.reachable.is_subset(&b.reachable) {
            Some(Ordering::Less)
        } else {
            None
        }
    }
}

impl Bound for Optw {
    type State = OptwState;
    type CostType = i32;

    fn get_dual_bound(&self, state: &Self::State) -> Option<i32> {
        if state.time > self.instance.closing[0] {
            return None;
        }
        if state.current == self.instance.vertices.len() || state.reachable.is_clear() {
            return Some(0);
        }
        let total_profit = state
            .reachable
            .ones()
            .map(|i| self.instance.profits[i].max(0))
            .sum::<i32>();
        let from = self.knapsack_bound(
            state,
            self.instance.closing[0] - state.time - self.min_from[state.current],
            &self.sorted_from,
        );
        let to = self.knapsack_bound(
            state,
            self.instance.closing[0] - state.time - self.min_to[0],
            &self.sorted_to,
        );
        Some(total_profit.min(from).min(to))
    }
}
