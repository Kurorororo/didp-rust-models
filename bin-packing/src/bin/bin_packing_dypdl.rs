use bin_packing::{Args, Instance, SolverChoice};
use clap::Parser;
use dypdl::prelude::*;
use dypdl_heuristic_search::{
    BeamSearchParameters, CabsParameters, FEvaluatorType, Parameters, create_caasdy,
    create_dual_bound_cabs, create_dual_bound_cahdbs2,
};
use rpid::timer::Timer;
use std::{rc::Rc, sync::Arc};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let instance = Instance::read_from_file(&args.input_file).unwrap();

    let mut model = Model::default();

    let n = instance.weights.len();
    let item = model.add_object_type("item", n).unwrap();

    let remaining = model
        .add_integer_resource_variable("remaining", false, 0)
        .unwrap();
    let unpacked = (0..n).collect::<Vec<_>>();
    let unpacked = model.create_set(item, &unpacked).unwrap();
    let unpacked = model.add_set_variable("unpacked", item, unpacked).unwrap();
    let bin_number = model
        .add_element_resource_variable("bin_number", item, true, 0)
        .unwrap();

    let weights = model
        .add_table_1d("weights", instance.weights.clone())
        .unwrap();
    let x = model.add_local_variable("x").unwrap();
    let no_available_item = model
        .add_boolean_state_function(
            "no available item",
            unpacked.all(
                x,
                Condition::comparison_i(ComparisonOperator::Gt, weights.element(x), remaining),
            ),
        )
        .unwrap();

    for (i, &wi) in instance.weights.iter().enumerate() {
        let mut pack = Transition::new(format!("{i}"));
        pack.set_cost(IntegerExpression::Cost);

        pack.add_effect(remaining, remaining - wi).unwrap();
        pack.add_effect(unpacked, unpacked.remove(i)).unwrap();

        pack.add_precondition(unpacked.contains(i));
        pack.add_precondition(Condition::comparison_i(
            ComparisonOperator::Le,
            wi,
            remaining,
        ));
        pack.add_precondition(Condition::comparison_e(
            ComparisonOperator::Le,
            bin_number,
            i + 1,
        ));

        model.add_forward_transition(pack).unwrap();

        let mut open_and_pack = Transition::new(format!("{i}"));
        open_and_pack.set_cost(1 + IntegerExpression::Cost);

        open_and_pack
            .add_effect(remaining, instance.capacity - wi)
            .unwrap();
        open_and_pack
            .add_effect(unpacked, unpacked.remove(i))
            .unwrap();
        open_and_pack
            .add_effect(bin_number, bin_number + 1)
            .unwrap();

        open_and_pack.add_precondition(unpacked.contains(i));
        open_and_pack.add_precondition(Condition::comparison_e(
            ComparisonOperator::Le,
            bin_number,
            i,
        ));
        open_and_pack.add_precondition(no_available_item.clone());

        model.add_forward_forced_transition(open_and_pack).unwrap();
    }

    model.add_base_case(vec![unpacked.is_empty()]).unwrap();

    model
        .add_dual_bound(IntegerExpression::ceil(
            ContinuousExpression::from(weights.sum(unpacked) - remaining)
                / ContinuousExpression::from(instance.capacity),
        ))
        .unwrap();

    let lb2_weight1 = instance
        .weights
        .iter()
        .map(|&x| if 2 * x > instance.capacity { 1 } else { 0 })
        .collect();
    let lb2_weight1 = model.add_table_1d("lb2_weight1", lb2_weight1).unwrap();
    let lb2_weight2 = instance
        .weights
        .iter()
        .map(|&x| if 2 * x == instance.capacity { 0.5 } else { 0.0 })
        .collect();
    let lb2_weight2 = model.add_table_1d("lb2_weight2", lb2_weight2).unwrap();
    let remaining_ge_half =
        Condition::comparison_i(ComparisonOperator::Ge, 2 * remaining, instance.capacity);
    model
        .add_dual_bound(
            lb2_weight1.sum(unpacked) + IntegerExpression::ceil(lb2_weight2.sum(unpacked))
                - IfThenElse::<IntegerExpression>::if_then_else(remaining_ge_half, 1, 0),
        )
        .unwrap();

    let lb3_weight = instance
        .weights
        .iter()
        .map(|&x| {
            if 3 * x > 2 * instance.capacity {
                1.0
            } else if 3 * x == 2 * instance.capacity {
                0.666
            } else if 3 * x > instance.capacity {
                0.5
            } else if 3 * x == instance.capacity {
                0.333
            } else {
                0.0
            }
        })
        .collect();
    let lb3_weight = model.add_table_1d("lb3_weight", lb3_weight).unwrap();
    let remaining_ge_one_third =
        Condition::comparison_i(ComparisonOperator::Ge, 3 * remaining, instance.capacity);
    model
        .add_dual_bound(
            IntegerExpression::ceil(lb3_weight.sum(unpacked))
                - IfThenElse::<IntegerExpression>::if_then_else(remaining_ge_one_third, 1, 0),
        )
        .unwrap();

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
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());

            if args.threads.get() > 1 {
                let model = Arc::new(model);
                create_dual_bound_cahdbs2(
                    model,
                    parameters,
                    FEvaluatorType::Plus,
                    args.threads.get(),
                )
            } else {
                let model = Rc::new(model);
                create_dual_bound_cabs(model, parameters, FEvaluatorType::Plus)
            }
        }
        SolverChoice::Astar => {
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());

            let model = Rc::new(model);
            create_caasdy(model, parameters, FEvaluatorType::Plus)
        }
    };

    let solution =
        io_util::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap();
    io_util::print_solution_statistics(&solution);

    if let Some(cost) = solution.cost {
        let sequence = solution
            .transitions
            .iter()
            .map(|t| t.get_full_name().parse().unwrap())
            .collect::<Vec<_>>();
        instance.print_solution(&sequence);

        if instance.validate(&sequence, cost) {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
