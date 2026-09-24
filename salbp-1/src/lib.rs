use clap::{Parser, ValueEnum};
use fixedbitset::FixedBitSet;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Clone, Debug)]
pub struct Instance {
    pub cycle_time: i32,
    pub task_times: Vec<i32>,
    pub predecessors: Vec<FixedBitSet>,
}

impl Instance {
    pub fn read_from_file(filename: &str) -> Result<Self, Box<dyn Error>> {
        let file = File::open(filename)?;
        let mut lines = BufReader::new(file).lines();
        let mut n = None;
        let mut cycle_time = None;
        let mut task_times = Vec::new();
        let mut predecessors = Vec::new();

        let mut line = lines.next().ok_or("empty file")??;

        loop {
            let trimmed = line.trim();

            if trimmed == "<end>" {
                break;
            }

            if line == "<precedence relations>" {
                let n = n.ok_or("missing number of tasks")?;
                predecessors = vec![FixedBitSet::with_capacity(n); n];

                line = lines.next().ok_or("unexpected EOF")??;
                let mut pair = line.trim().split(",").collect::<Vec<_>>();

                while pair.len() == 2 {
                    let i = pair[0].parse::<usize>()? - 1;
                    let j = pair[1].parse::<usize>()? - 1;
                    predecessors[j].insert(i);

                    line = lines.next().ok_or("unexpected EOF")??;
                    pair = line.split(",").collect::<Vec<_>>();
                }
            } else {
                if line == "<number of tasks>" {
                    let line = lines.next().ok_or("unexpected EOF")??;
                    n = Some(line.trim().parse()?);
                }

                if line == "<cycle time>" {
                    let line = lines.next().ok_or("unexpected EOF")??;
                    cycle_time = Some(line.trim().parse()?);
                }

                if line == "<task times>" {
                    let n = n.ok_or("missing number of tasks")?;
                    task_times = vec![None; n];

                    for _ in 0..n {
                        let line = lines.next().ok_or("unexpected EOF")??;
                        let pair = line.split_whitespace().collect::<Vec<_>>();

                        if pair.len() != 2 {
                            return Err("invalid task time".into());
                        }

                        let task_id = pair[0].parse::<usize>()? - 1;
                        let task_time = pair[1].parse::<i32>()?;
                        task_times[task_id] = Some(task_time);
                    }
                }

                line = lines.next().ok_or("unexpected EOF")??;
            }
        }

        let cycle_time = cycle_time.ok_or("missing cycle time")?;
        let task_times = task_times
            .into_iter()
            .enumerate()
            .map(|(i, t)| t.ok_or(format!("missing task time for task {}", i + 1)))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            cycle_time,
            task_times,
            predecessors,
        })
    }

    pub fn validate(&self, solution: &[usize], cost: i32) -> bool {
        let mut remaining = 0;
        let mut scheduled = FixedBitSet::with_capacity(self.task_times.len());
        let mut recomputed_cost = 0;

        for &task in solution {
            if task >= self.task_times.len() {
                println!("Invalid task id: {task}");

                return false;
            }

            if scheduled.contains(task) {
                println!("Task {task} is already scheduled");

                return false;
            }

            let unsatisfied = self.predecessors[task]
                .difference(&scheduled)
                .map(|i| i + 1)
                .collect::<Vec<usize>>();

            if !unsatisfied.is_empty() {
                println!("Task {task} has unsatisfied predecessors: {unsatisfied:?}",);

                return false;
            }

            if self.task_times[task] > remaining {
                remaining = self.cycle_time;
                recomputed_cost += 1;
            }

            remaining -= self.task_times[task];
            scheduled.insert(task);
        }

        if recomputed_cost != cost {
            println!("Invalid cost: {cost} != {recomputed_cost}");

            return false;
        }

        if scheduled.count_ones(..) != self.task_times.len() {
            println!(
                "Tasks: {not_scheduled:?} are not scheduled",
                not_scheduled = scheduled.ones().map(|i| i + 1).collect::<Vec<_>>()
            );

            return false;
        }

        true
    }

    pub fn print_solution(&self, solution: &[usize]) {
        let mut stations = vec![];
        let mut remaining = 0;

        for &i in solution {
            if self.task_times[i] > remaining {
                stations.push(vec![]);
                remaining = self.cycle_time;
            }

            stations.last_mut().unwrap().push(i + 1);
            remaining -= self.task_times[i];
        }

        println!("Schedule: {stations:?}");
    }
}

#[derive(Debug, Clone, ValueEnum)]
pub enum SolverChoice {
    Cabs,
    Astar,
}

#[derive(Debug, Parser)]
pub struct Args {
    #[arg(
        short = 'j',
        long,
        default_value = "1",
        help = "Number of threads for CABS"
    )]
    pub threads: std::num::NonZeroUsize,
    #[arg(help = "Input file")]
    pub input_file: String,
    #[arg(short, long, value_enum, default_value_t = SolverChoice::Cabs, help = "Solver")]
    pub solver: SolverChoice,
    #[arg(long, default_value_t = String::from("history.csv"), help = "File to save the history")]
    pub history: String,
    #[arg(short, long, default_value_t = 1800.0, help = "Time limit")]
    pub time_limit: f64,
}
