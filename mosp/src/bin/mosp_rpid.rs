use clap::Parser;
use fixedbitset::FixedBitSet;
use mosp::{self, Args, SolverChoice};
use rpid::prelude::*;
use rpid::{io, solvers, timer::Timer};
use std::cmp;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Clone, Debug)]
struct Mosp {
    matrix: Vec<FixedBitSet>,
    transposed: Vec<FixedBitSet>,
    column_neighbors: Vec<FixedBitSet>,
}

impl From<Vec<FixedBitSet>> for Mosp {
    fn from(matrix: Vec<FixedBitSet>) -> Self {
        let transposed = mosp::transpose(&matrix);
        let column_neighbors = transposed
            .iter()
            .map(|column| {
                let mut set = FixedBitSet::with_capacity(column.len());
                column.ones().for_each(|i| set.union_with(&matrix[i]));

                set
            })
            .collect();

        Self {
            matrix,
            transposed,
            column_neighbors,
        }
    }
}

impl Mosp {
    fn reconstruct_solution(&self, indices: &[usize]) -> Vec<usize> {
        let mut solution = Vec::with_capacity(self.matrix.len());
        let mut produced = FixedBitSet::with_capacity(self.matrix.len());

        for i in indices.iter().flat_map(|&j| self.transposed[j].ones()) {
            if !produced.contains(i) {
                produced.insert(i);
                solution.push(i);
            }
        }

        solution
    }
}

#[derive(Clone)]
struct MospState {
    remaining: FixedBitSet,
    opened: FixedBitSet,
}

impl Dp for Mosp {
    type State = MospState;
    type CostType = i32;
    type Label = usize;

    fn get_target(&self) -> MospState {
        let mut remaining = FixedBitSet::with_capacity(self.transposed.len());
        remaining.insert_range(..);
        let opened = FixedBitSet::with_capacity(self.transposed.len());

        MospState { remaining, opened }
    }

    fn get_successors(
        &self,
        state: &Self::State,
    ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
        let mut currently_open = state.opened.clone();
        currently_open.intersect_with(&state.remaining);

        state.remaining.ones().map(move |i| {
            let mut remaining = state.remaining.clone();
            remaining.remove(i);

            let mut opened = state.opened.clone();
            opened.union_with(&self.column_neighbors[i]);

            let mut newly_open = self.column_neighbors[i].clone();
            newly_open.difference_with(&state.opened);
            let weight = currently_open.union(&newly_open).count() as i32;

            (MospState { remaining, opened }, weight, i)
        })
    }

    fn get_base_cost(&self, state: &Self::State) -> Option<Self::CostType> {
        if state.remaining.is_clear() {
            Some(0)
        } else {
            None
        }
    }

    fn combine_cost_weights(&self, a: Self::CostType, b: Self::CostType) -> Self::CostType {
        cmp::max(a, b)
    }
}

impl Dominance for Mosp {
    type State = MospState;
    type Key = FixedBitSet;

    fn get_key(&self, state: &Self::State) -> Self::Key {
        state.remaining.clone()
    }
}

impl Bound for Mosp {
    type State = MospState;
    type CostType = i32;

    fn get_dual_bound(&self, _: &Self::State) -> Option<Self::CostType> {
        Some(0)
    }
}

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let matrix = mosp::read_from_file(&args.input_file).unwrap();
    let mosp = Mosp::from(matrix);

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
                    mosp.clone(),
                    parameters,
                    cabs_parameters,
                    args.threads.get(),
                )
            } else {
                solvers::create_cabs(mosp.clone(), parameters, cabs_parameters)
            };
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::Astar => {
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_astar(mosp.clone(), parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
    };
    io::print_solution_statistics(&solution);

    if let Some(cost) = solution.cost {
        let schedule = mosp.reconstruct_solution(&solution.transitions);
        let transitions = schedule
            .iter()
            .map(|t| format!("{t}"))
            .collect::<Vec<_>>()
            .join(" ");
        println!("Schedule: {transitions}");

        if mosp::validate(&mosp.matrix, &schedule, cost) {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
