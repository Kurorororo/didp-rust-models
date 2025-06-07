use clap::Parser;
use dypdl::prelude::*;
use dypdl_heuristic_search::{
    create_caasdy, create_dual_bound_cabs, BeamSearchParameters, CabsParameters, FEvaluatorType,
    Parameters,
};
use fixedbitset::FixedBitSet;
use mosp::{self, Args, SolverChoice};
use rpid::timer::Timer;
use std::rc::Rc;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let matrix = mosp::read_from_file(&args.input_file).unwrap();

    let mut model = Model::default();

    let transposed = mosp::transpose(&matrix);
    let n = transposed.len();
    let customer = model.add_object_type("customer", n).unwrap();

    let column_neighbors = transposed
        .iter()
        .map(|column| {
            let mut set = FixedBitSet::with_capacity(column.len());
            column.ones().for_each(|i| set.union_with(&matrix[i]));
            let set = set.ones().collect::<Vec<_>>();

            model.create_set(customer, &set).unwrap()
        })
        .collect::<Vec<_>>();

    let remaining = (0..n).collect::<Vec<_>>();
    let remaining = model.create_set(customer, &remaining).unwrap();
    let remaining = model
        .add_set_variable("remaining", customer, remaining)
        .unwrap();
    let opened = vec![];
    let opened = model.create_set(customer, &opened).unwrap();
    let opened = model.add_set_variable("opened", customer, opened).unwrap();

    for (i, neighbors) in column_neighbors.iter().enumerate() {
        let mut close = Transition::new(format!("{}", i));
        close.set_cost(IntegerExpression::max(
            ((opened & remaining) | (neighbors.clone() - opened)).len(),
            IntegerExpression::Cost,
        ));
        close.add_effect(remaining, remaining.remove(i)).unwrap();
        close
            .add_effect(opened, opened | neighbors.clone())
            .unwrap();
        close.add_precondition(remaining.contains(i));

        model.add_forward_transition(close).unwrap();
    }

    model.add_base_case(vec![remaining.is_empty()]).unwrap();

    model.add_dual_bound(IntegerExpression::from(0)).unwrap();

    let model = Rc::new(model);

    let parameters = Parameters::<i32> {
        time_limit: Some(args.time_limit),
        ..Default::default()
    };

    let mut solver = match args.solver {
        SolverChoice::Cabs => {
            let beam_search_parameters = BeamSearchParameters {
                parameters,
                ..Default::default()
            };
            let parameters = CabsParameters {
                beam_search_parameters,
                ..Default::default()
            };
            println!("Preparing time: {}s", timer.get_elapsed_time());

            create_dual_bound_cabs(model, parameters, FEvaluatorType::Max)
        }
        SolverChoice::Astar => {
            println!("Preparing time: {}s", timer.get_elapsed_time());

            create_caasdy(model, parameters, FEvaluatorType::Max)
        }
    };

    let solution =
        io_util::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap();
    io_util::print_solution_statistics(&solution);

    if let Some(cost) = solution.cost {
        let indices = solution
            .transitions
            .iter()
            .map(|t| t.get_full_name().parse().unwrap())
            .collect::<Vec<usize>>();
        let mut sequence = Vec::with_capacity(matrix.len());
        let mut produced = FixedBitSet::with_capacity(matrix.len());

        for i in indices.iter().flat_map(|&j| transposed[j].ones()) {
            if !produced.contains(i) {
                produced.insert(i);
                sequence.push(i);
            }
        }
        println!(
            "Schedule: {}",
            sequence
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        );

        if mosp::validate(&matrix, &sequence, cost) {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
