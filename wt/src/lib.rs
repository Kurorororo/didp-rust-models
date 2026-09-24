use clap::{Parser, ValueEnum};
use fixedbitset::FixedBitSet;
use rpid::io;
use std::cmp;
use std::collections::VecDeque;
use std::error::Error;
use std::fs;

#[derive(Clone, Debug)]
pub struct Instance {
    pub processing_times: Vec<i32>,
    pub deadlines: Vec<i32>,
    pub weights: Vec<i32>,
}

impl Instance {
    pub fn read_from_file(filename: &str) -> Result<Self, Box<dyn Error>> {
        let file = fs::read_to_string(filename)?;
        let mut digits = file.split_whitespace();

        let n = digits
            .next()
            .ok_or("failed to parse the number of jobs".to_owned())?
            .parse()?;
        let matrix = io::read_matrix(&mut digits, n, 3)?;
        let processing_time = matrix.iter().map(|row| row[0]).collect();
        let deadline = matrix.iter().map(|row| row[1]).collect();
        let weights = matrix.iter().map(|row| row[2]).collect();

        Ok(Self {
            processing_times: processing_time,
            deadlines: deadline,
            weights,
        })
    }

    pub fn validate(&self, solution: &[usize], cost: i32) -> bool {
        let n = self.processing_times.len();

        if solution.len() != n {
            println!("Invalid solution length: {} != {}", solution.len(), n);

            return false;
        }

        let mut time = 0;
        let mut scheduled = vec![false; n];
        let mut recomputed_cost = 0;

        for &j in solution {
            if j >= n {
                println!("Invalid job index: {j}");

                return false;
            }

            if scheduled[j] {
                println!("Job {j} is scheduled more than once");

                return false;
            }

            time += self.processing_times[j];
            recomputed_cost += self.weights[j] * cmp::max(0, time - self.deadlines[j]);
            scheduled[j] = true;
        }

        if recomputed_cost != cost {
            println!("Invalid cost: {cost} != {recomputed_cost}");

            return false;
        }

        true
    }

    fn has_path(i: usize, j: usize, successors: &[FixedBitSet]) -> bool {
        let mut open = VecDeque::from([i]);
        let mut checked = vec![false; successors.len()];

        while let Some(u) = open.pop_front() {
            for v in successors[u].ones() {
                if v == j {
                    return true;
                }

                if !checked[v] {
                    open.push_back(v);
                    checked[v] = true;
                }
            }
        }

        false
    }

    fn check_kanet_conditions(
        &self,
        i: usize,
        predecessors_i: &FixedBitSet,
        successors_i: &FixedBitSet,
        j: usize,
        predecessors_j: &FixedBitSet,
        successors_j: &FixedBitSet,
    ) -> bool {
        let p_i = self.processing_times[i];
        let d_i = self.deadlines[i];
        let w_i = self.weights[i];
        let p_j = self.processing_times[j];
        let d_j = self.deadlines[j];
        let w_j = self.weights[j];

        let p_predecessors_j = predecessors_j
            .ones()
            .map(|k| self.processing_times[k])
            .sum::<i32>();
        let p_common = predecessors_i
            .union(predecessors_j)
            .map(|k| self.processing_times[k])
            .sum::<i32>()
            + p_i
            + p_j;

        let k1_common = (w_i - w_j) * p_common;

        if p_i <= p_j
            && w_i >= w_j
            && (w_i * d_i <= cmp::max(w_i * d_j, k1_common + w_j * d_j)
                || w_i * d_i <= k1_common + w_j * (p_predecessors_j + p_j))
        {
            return true;
        }

        let p_not_successors_i = successors_i
            .zeroes()
            .map(|k| self.processing_times[k])
            .sum::<i32>();
        let mut successors_i_or_j = successors_i.clone();
        successors_i_or_j.union_with(successors_j);
        let p_not_successors_i_or_j = successors_i_or_j
            .zeroes()
            .map(|k| self.processing_times[k])
            .sum::<i32>();

        let k2_common = (w_j - w_i) * p_not_successors_i;

        if p_i <= p_j
            && w_i < w_j
            && w_j * d_j >= k2_common + w_i * d_i
            && w_j * d_j >= k2_common + w_i * (p_not_successors_i_or_j - p_j)
        {
            return true;
        }

        if p_i <= p_j
            && w_i < w_j
            && w_i * d_i <= (w_i - w_j) * p_not_successors_i + w_j * (p_predecessors_j + p_j)
            && w_i * p_i <= (w_i - w_j) * (p_not_successors_i - p_predecessors_j) + w_j * p_j
        {
            return true;
        }

        if w_i >= w_j
            && w_j * d_j >= cmp::min(w_j * d_i, (w_j - w_i) * p_common + w_i * d_i)
            && w_j * d_j >= w_j * p_not_successors_i - w_i * p_j
        {
            return true;
        }

        if w_i >= w_j
            && w_i * d_i <= (w_i - w_j) * p_common + w_j * (p_predecessors_j + p_j)
            && w_i * p_j >= w_j * (p_not_successors_i - p_predecessors_j - p_j)
        {
            return true;
        }

        if w_i < w_j
            && w_j * d_j >= (w_j - w_i) * p_not_successors_i + w_i * d_i
            && w_j * d_j >= w_j * p_not_successors_i - w_i * p_j
        {
            return true;
        }

        if d_j >= p_not_successors_i {
            return true;
        }

        false
    }

    pub fn extract_precedence(&self) -> (Vec<FixedBitSet>, Vec<FixedBitSet>) {
        let n = self.processing_times.len();
        let mut predecessors = vec![FixedBitSet::with_capacity(n); n];
        let mut successors = vec![FixedBitSet::with_capacity(n); n];
        let mut change = true;

        while change {
            change = false;

            for i in 0..n {
                for j in 0..n {
                    if i != j
                        && !predecessors[j].contains(i)
                        && !Self::has_path(j, i, &successors)
                        && self.check_kanet_conditions(
                            i,
                            &predecessors[i],
                            &successors[i],
                            j,
                            &predecessors[j],
                            &successors[j],
                        )
                    {
                        predecessors[j].insert(i);
                        successors[i].insert(j);
                        change = true;
                    }
                }
            }
        }

        (predecessors, successors)
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
