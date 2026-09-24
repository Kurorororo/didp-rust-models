use clap::Parser;
use fixedbitset::FixedBitSet;
use m_pdtsp::{Args, RoundedInstance, SolverChoice};
use rpid::prelude::*;
use rpid::{algorithms, io, solvers, timer::Timer};
use std::cmp::Ordering;
use tsplib_parser::Instance;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Clone)]
struct OnePdtsp {
    capacity: i32,
    demands: Vec<i32>,
    predecessors: Vec<FixedBitSet>,
    distances: Vec<Vec<Option<i32>>>,
    sorted_edges: Vec<(usize, usize, i32)>,
    node_to_sorted_out_edges: Vec<Vec<(usize, i32)>>,
    sorted_edges_to_goal: Vec<(usize, i32)>,
}

impl From<RoundedInstance> for OnePdtsp {
    fn from(instance: RoundedInstance) -> Self {
        let (predecessors, distances) = instance.extract_predecessors_and_filtered_distances();
        let demands = instance.demands.iter().map(|d| d.iter().sum()).collect();
        let sorted_edges = algorithms::sort_weight_matrix_with_option(&distances);
        let n = instance.nodes.len();
        let mut node_to_sorted_out_edges = vec![Vec::new(); n];
        let mut sorted_edges_to_goal = Vec::new();

        for &(i, j, w) in &sorted_edges {
            node_to_sorted_out_edges[i].push((j, w));

            if j == n - 1 {
                sorted_edges_to_goal.push((i, w));
            }
        }

        Self {
            capacity: instance.capacity,
            demands,
            predecessors,
            distances,
            sorted_edges,
            node_to_sorted_out_edges,
            sorted_edges_to_goal,
        }
    }
}

#[derive(Clone)]
struct OnePdtspState {
    unvisited: FixedBitSet,
    current: usize,
    load: i32,
}

impl Dp for OnePdtsp {
    type State = OnePdtspState;
    type CostType = i32;
    type Label = usize;

    fn get_target(&self) -> Self::State {
        let n = self.demands.len();
        let mut unvisited = FixedBitSet::with_capacity(n);
        unvisited.insert_range(1..n - 1);

        OnePdtspState {
            unvisited,
            current: 0,
            load: 0,
        }
    }

    fn get_successors(
        &self,
        state: &Self::State,
    ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
        state.unvisited.ones().filter_map(|next| {
            if let Some(d) = self.distances[state.current][next] {
                let load = state.load + self.demands[next];

                if load <= self.capacity && state.unvisited.is_disjoint(&self.predecessors[next]) {
                    let mut unvisited = state.unvisited.clone();
                    unvisited.remove(next);
                    let successor = OnePdtspState {
                        unvisited,
                        current: next,
                        load,
                    };

                    Some((successor, d, next))
                } else {
                    None
                }
            } else {
                None
            }
        })
    }

    fn get_base_cost(&self, state: &Self::State) -> Option<Self::CostType> {
        if state.unvisited.is_clear() {
            self.distances[state.current][self.demands.len() - 1]
        } else {
            None
        }
    }
}

impl Dominance for OnePdtsp {
    type State = OnePdtspState;
    type Key = (FixedBitSet, usize);

    fn get_key(&self, state: &Self::State) -> Self::Key {
        (state.unvisited.clone(), state.current)
    }

    fn compare(&self, a: &Self::State, b: &Self::State) -> Option<Ordering> {
        Some(b.load.cmp(&a.load))
    }
}

impl Bound for OnePdtsp {
    type State = OnePdtspState;
    type CostType = i32;

    fn get_dual_bound(&self, state: &Self::State) -> Option<Self::CostType> {
        let n = state.unvisited.count_ones(..);
        let goal = self.demands.len() - 1;

        if n == 0 {
            return self.distances[state.current][goal];
        }

        let minimum_start = self.node_to_sorted_out_edges[state.current]
            .iter()
            .find_map(|&(i, w)| {
                if state.unvisited.contains(i) {
                    Some(w)
                } else {
                    None
                }
            })?;

        let iter = self
            .sorted_edges
            .iter()
            .filter(|(i, j, _)| state.unvisited.contains(*i) && state.unvisited.contains(*j))
            .copied();
        let mst_weight = algorithms::compute_minimum_spanning_tree_weight(goal - 1, n, iter);

        let minimum_return = self.sorted_edges_to_goal.iter().find_map(|&(i, w)| {
            if state.unvisited.contains(i) {
                Some(w)
            } else {
                None
            }
        })?;

        Some(minimum_start + mst_weight + minimum_return)
    }
}

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let filepath = args.input_file;

    let instance = Instance::load(&filepath).unwrap();
    let instance = RoundedInstance::try_from(instance).unwrap();
    let one_pdtsp = OnePdtsp::from(instance.clone());

    let parameters = SearchParameters {
        time_limit: Some(args.time_limit),
        ..Default::default()
    };
    let solution = match args.solver {
        SolverChoice::Cabs => {
            let cabs_parameters = CabsParameters::default();
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = if args.threads.get() > 1 {
                solvers::create_parallel_cabs(
                    one_pdtsp,
                    parameters,
                    cabs_parameters,
                    args.threads.get(),
                )
            } else {
                solvers::create_cabs(one_pdtsp, parameters, cabs_parameters)
            };
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::Astar => {
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_astar(one_pdtsp, parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
    };
    io::print_solution_statistics(&solution);

    if let Some(cost) = solution.cost {
        instance.print_solution(&solution.transitions);

        if instance.validate(&solution.transitions, cost) {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
