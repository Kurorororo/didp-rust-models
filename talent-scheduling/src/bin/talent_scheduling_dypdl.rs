use clap::Parser;
use dypdl::prelude::*;
use dypdl_heuristic_search::{
    BeamSearchParameters, CabsParameters, FEvaluatorType, Parameters, create_caasdy,
    create_dual_bound_cabs, create_dual_bound_cahdbs2,
};
use rpid::timer::Timer;
use std::{rc::Rc, sync::Arc};
use talent_scheduling::{Args, Instance, SolverChoice};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let instance = Instance::read_from_file(&args.input_file).unwrap();
    let (simplified_instance, single_actor_cost, scene_to_originals) = instance.simplify();

    let mut model = Model::default();

    let n = simplified_instance.scene_to_duration.len();
    let scene = model.add_object_type("scene", n).unwrap();
    let m = std::cmp::max(simplified_instance.actor_to_cost.len(), 1);
    let actor = model.add_object_type("actor", m).unwrap();

    let remaining = (0..n).collect::<Vec<_>>();
    let remaining = model.create_set(scene, &remaining).unwrap();
    let remaining = model
        .add_set_variable("remaining", scene, remaining)
        .unwrap();

    let scene_to_actors = simplified_instance.create_scene_to_actors();
    let mut scene_to_subsumption_candidates = vec![vec![]; n];

    for i in 0..n {
        for j in 0..n {
            if i != j && scene_to_actors[i].is_subset(&scene_to_actors[j]) {
                scene_to_subsumption_candidates[i].push(j);
            }
        }
    }

    let scene_to_base_cost = scene_to_actors
        .iter()
        .enumerate()
        .map(|(i, actors)| {
            actors
                .ones()
                .map(|j| simplified_instance.actor_to_cost[j])
                .sum::<i32>()
                * simplified_instance.scene_to_duration[i]
        })
        .collect();
    let scene_to_base_cost = model
        .add_table_1d("scene to base cost", scene_to_base_cost)
        .unwrap();
    let scene_to_actors = scene_to_actors
        .iter()
        .map(|actors| {
            let set = actors.ones().collect::<Vec<_>>();
            model.create_set(actor, &set).unwrap()
        })
        .collect::<Vec<_>>();
    let scene_to_actors = model
        .add_table_1d("scene to actors", scene_to_actors)
        .unwrap();
    let mut actor_costs = simplified_instance.actor_to_cost.clone();
    actor_costs.resize(m, 0);
    let actor_cost = model.add_table_1d("actor cost", actor_costs).unwrap();

    let arrived = model
        .add_set_state_function("arrived", scene_to_actors.union(m, !remaining))
        .unwrap();
    let on_location = model
        .add_set_state_function(
            "on_location",
            scene_to_actors.union(m, remaining) & arrived.clone(),
        )
        .unwrap();

    for i in 0..n {
        let mut actor_equivalent_shoot = Transition::new(format!("{i}"));
        actor_equivalent_shoot.set_cost(scene_to_base_cost.element(i) + IntegerExpression::Cost);
        actor_equivalent_shoot
            .add_effect(remaining, remaining.remove(i))
            .unwrap();
        actor_equivalent_shoot.add_precondition(remaining.contains(i));
        actor_equivalent_shoot
            .add_precondition(scene_to_actors.element(i).is_equal(on_location.clone()));

        model
            .add_forward_forced_transition(actor_equivalent_shoot)
            .unwrap();
    }

    let mut transition_ids = Vec::with_capacity(n);
    let mut state_functions = Vec::with_capacity(n);

    for (i, &d) in simplified_instance.scene_to_duration.iter().enumerate() {
        let on_location_i = scene_to_actors.element(i) | on_location.clone();

        let mut shoot = Transition::new(format!("{i}"));
        shoot.set_cost(d * actor_cost.sum(on_location_i) + IntegerExpression::Cost);
        shoot.add_effect(remaining, remaining.remove(i)).unwrap();
        shoot.add_precondition(remaining.contains(i));

        let id = model.add_forward_transition(shoot).unwrap();
        transition_ids.push(id);

        let state_function = model
            .add_set_state_function(
                format!("arrived_or_playing_in_{i}"),
                arrived.clone() | scene_to_actors.element(i),
            )
            .unwrap();
        state_functions.push(state_function);
    }

    for i in 0..n {
        for &j in &scene_to_subsumption_candidates[i] {
            model
                .add_transition_dominance_with_conditions(
                    &transition_ids[j],
                    &transition_ids[i],
                    vec![
                        scene_to_actors
                            .element(j)
                            .is_subset(state_functions[i].clone()),
                    ],
                )
                .unwrap();
        }
    }

    model
        .add_base_case_with_cost(
            vec![remaining.is_empty()],
            IntegerExpression::from(single_actor_cost),
        )
        .unwrap();

    model
        .add_dual_bound(scene_to_base_cost.sum(remaining) + single_actor_cost)
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
        let schedule = solution
            .transitions
            .iter()
            .map(|t| t.get_full_name().parse::<usize>().unwrap())
            .flat_map(|i| scene_to_originals[i].iter().cloned())
            .collect::<Vec<_>>();
        println!(
            "Schedule: {}",
            schedule
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        );

        if instance.validate(&schedule, cost) {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
