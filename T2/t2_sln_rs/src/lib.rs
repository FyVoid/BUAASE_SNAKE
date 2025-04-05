use std::collections::{HashMap, HashSet, VecDeque};

use wasm_bindgen::prelude::*;

mod common;
// #[cfg(all(test, not(target_arch = "wasm32")))]
// mod verifier;

use common::*;

#[wasm_bindgen]
pub fn greedy_snake_move_barriers(pos: &[i32], food: &[i32], barriers: &[i32]) -> i32 {
    let head = Point::new(pos[0], pos[1]);
    let mut board = make_board::<8>(food, barriers);
    let init_dir = match pos[0] - pos[2] {
        1 => Direction::R,
        -1 => Direction::L,
        _ => match pos[1] - pos[3] {
            1 => Direction::U,
            -1 => Direction::D,
            _ => unreachable!(),
        },
    };
    if !reachable(&mut board, &head, init_dir) {
        return -1;
    }
    // find the nearest food
    let mut foods = vec![];
    for i in 1..=8 {
        for j in 1..=8 {
            let point = Point::new(i, j);
            if let Some(Tile::Food(depth)) = board.get_tile(point) {
                foods.push((point, depth));
            }
        }
    }
    foods.sort_by(|a, b| a.1.cmp(&b.1));

    if foods.is_empty() {
        panic!("No food found");
    }

    let path = pathfinding(&mut board, &head, init_dir, &foods[0].0);
    if let Some(dir) = path {
        return match dir {
            Direction::U => 0,
            Direction::L => 1,
            Direction::D => 2,
            Direction::R => 3,
        };
    }

    // choose random available direction
    for &dir in &[Direction::U, Direction::L, Direction::D, Direction::R] {
        if matches!(
            (init_dir, dir),
            (Direction::U, Direction::D)
                | (Direction::D, Direction::U)
                | (Direction::L, Direction::R)
                | (Direction::R, Direction::L)
        ) {
            continue; // Snake can't go back
        }
        let next = head.step(dir);
        if board.inbounds(next) {
            if let Some(tile) = board.get_tile(next) {
                match tile {
                    Tile::Empty(_) | Tile::Food(_) => {
                        return match dir {
                            Direction::U => 0,
                            Direction::L => 1,
                            Direction::D => 2,
                            Direction::R => 3,
                        };
                    }
                    _ => {}
                }
            }
        }
    }

    panic!("No valid move found")
}

fn make_board<const N: usize>(food: &[i32], barriers: &[i32]) -> Board<N> {
    let mut board = Board::<N>::new();
    food.chunks(2).for_each(|chunk| {
        board.set_tile(chunk, Tile::Food(usize::MAX));
    });
    barriers.chunks(2).for_each(|chunk| {
        board.set_tile(chunk, Tile::Barrier);
    });
    board
}

fn make_board_printing<const N: usize>(pos: &[i32], food: &[i32], barriers: &[i32]) -> Board<N> {
    let mut board = Board::<N>::new();
    let head = Point::new(pos[0], pos[1]);
    board.set_tile(head, Tile::SnakeHead);
    pos.chunks(2).skip(1).for_each(|chunk| {
        board.set_tile(chunk, Tile::SnakeBody);
    });
    food.chunks(2).for_each(|chunk| {
        board.set_tile(chunk, Tile::Food(usize::MAX));
    });
    barriers.chunks(2).for_each(|chunk| {
        board.set_tile(chunk, Tile::Barrier);
    });
    board
}

fn reachable<const N: usize>(board: &mut Board<N>, start: &Point, init_dir: Direction) -> bool {
    use std::collections::VecDeque;
    let mut queue = VecDeque::new();
    let mut visited = HashMap::new();
    queue.push_back((*start, init_dir, 0));
    visited.insert(*start, DirectionMarker::empty());

    while let Some((current, first_dir, depth)) = queue.pop_front() {
        if let Some(v) = visited.get_mut(&current) {
            if v.contains(first_dir) {
                continue;
            } else {
                v.set(first_dir);
            }
        } else {
            visited.insert(current, DirectionMarker::from_dir(first_dir));
        }
        for &dir in &[Direction::U, Direction::L, Direction::D, Direction::R] {
            match (first_dir, dir) {
                (Direction::U, Direction::D)
                | (Direction::D, Direction::U)
                | (Direction::L, Direction::R)
                | (Direction::R, Direction::L) => {
                    continue; // Snake can't go back
                }
                _ => {}
            }
            let next = current.step(dir);
            if let Some(tile) = board.get_tile(next) {
                match tile {
                    Tile::Empty(d0) => {
                        queue.push_back((next, dir, depth + 1));
                        if depth + 1 < d0 {
                            board.set_tile(next, Tile::Empty(depth + 1));
                        }
                    }
                    Tile::Food(d0) => {
                        queue.push_back((next, dir, depth + 1));
                        if depth + 1 < d0 {
                            board.set_tile(next, Tile::Food(depth + 1));
                        }
                    }
                    Tile::Barrier => {
                        continue;
                    }
                    Tile::SnakeHead | Tile::SnakeBody => {
                        unreachable!();
                    }
                }
            }
        }
    }

    let mut dead_end_food_count = 0;
    for i in 1..=N {
        for j in 1..=N {
            let point = Point::new(i as i32, j as i32);
            if let Some(Tile::Food(_)) = board.get_tile(point) {
                if !visited.contains_key(&point) {
                    return false;
                }
                if visited[&point].count() == 1 {
                    dead_end_food_count += 1;
                }
            }
        }
    }
    if dead_end_food_count > 1 {
        return false;
    }
    true
}

fn pathfinding<const N: usize>(
    board: &mut Board<N>,
    source: &Point,
    init_dir: Direction,
    target: &Point,
) -> Option<Direction> {
    let mut queue: VecDeque<Point> = VecDeque::new();
    let mut visited: HashSet<Point> = HashSet::new();

    let mut predecessor: HashMap<Point, (Point, Direction)> = HashMap::new();

    if source == target {
        return None;
    }

    queue.push_back(*source);
    visited.insert(*source);

    while let Some(current_point) = queue.pop_front() {
        for &dir in &[Direction::U, Direction::L, Direction::D, Direction::R] {
            if current_point == *source {
                let opposite_init_dir = match init_dir {
                    Direction::U => Direction::D,
                    Direction::D => Direction::U,
                    Direction::L => Direction::R,
                    Direction::R => Direction::L,
                };
                if dir == opposite_init_dir {
                    continue;
                }
            }

            let next_point = current_point.step(dir);

            let is_passable = matches!(
                board.get_tile(next_point),
                Some(Tile::Empty(_)) | Some(Tile::Food(_))
            );
            if !is_passable {
                continue;
            }

            if visited.contains(&next_point) {
                continue;
            }

            visited.insert(next_point);
            predecessor.insert(next_point, (current_point, dir));
            queue.push_back(next_point);

            if next_point == *target {
                let mut trace = next_point;
                loop {
                    let (prev_point, step_dir) = predecessor[&trace];

                    if prev_point == *source {
                        return Some(step_dir);
                    }
                    trace = prev_point;
                }
            }
        }
    }
    None
}

// #[cfg(all(test, not(target_arch = "wasm32")))]
// mod tests {
//     use std::io::Write;

//     use crate::verifier::SnakeState;

//     use super::*;

//     #[test]
//     fn testcase1() {
//         let pos = [1, 5, 1, 4, 1, 3, 1, 2]
//             .chunks(2)
//             .map(|v| Point::new(v[0], v[1]))
//             .collect::<Vec<_>>();
//         let food = [1, 1]
//             .chunks(2)
//             .map(|v| Point::new(v[0], v[1]))
//             .collect::<Vec<_>>();
//         let barriers = [
//             2, 1, 2, 2, 2, 3, 2, 4, 2, 5, 2, 6, 2, 7, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1, 7, 2,
//         ]
//         .chunks(2)
//         .map(|v| Point::new(v[0], v[1]))
//         .collect::<Vec<_>>();

//         let mut game = SnakeState::new(&pos, &food, &barriers);
//         loop {
//             let (pos, food, bar) = game.get_pair_vec();
//             let dir = greedy_snake_move_barriers(&pos, &food, &bar);
//             //print
//             // println!("Direction: {:?}", dir);
//             make_board_printing::<8>(&pos, &food, &bar).pretty_print();
//             if dir == -1 {
//                 println!("Unreachable");
//                 break;
//             } else {
//                 let dir = match dir {
//                     0 => Direction::U,
//                     1 => Direction::L,
//                     2 => Direction::D,
//                     3 => Direction::R,
//                     _ => unreachable!(),
//                 };
//                 let r = game.step(dir);
//                 if r.is_ok() {
//                     if r.unwrap() {
//                         assert!(game.get_pair_vec().1.is_empty());
//                         println!("Game over");
//                         break;
//                     }
//                 } else {
//                     println!("Error: {:?}", r);
//                     break;
//                 }
//             }
//         }
//     }

//     #[link(name = "a")]
//     unsafe extern "C" {
//         unsafe fn _func_t2(pos: *const i32, food: *const i32, barriers: *const i32) -> i32;
//     }

//     #[test]
//     fn fuzz_snake_movement() {
//         let mut success_count = 0;
//         let mut unreachable_count = 0;
//         for _ in 0..10000 {
//             let mut steps = 0;
//             let mut game = SnakeState::random(4, 1, 12);

//             let (pos, food, bar) = game.get_pair_vec();
//             std::io::stdout().flush().unwrap();
//             println!("New game board:");
//             make_board_printing::<8>(&pos, &food, &bar).pretty_print();

//             loop {
//                 let (pos, food, bar) = game.get_pair_vec();
//                 // let dir = greedy_snake_move_barriers(&pos, &food, &bar);
//                 let dir = unsafe { _func_t2(pos.as_ptr(), food.as_ptr(), bar.as_ptr()) };
//                 if dir == -1 {
//                     unreachable_count += 1;
//                     println!("Unreachable, manually check. Step: {}", steps);
//                     break;
//                 } else {
//                     let dir = match dir {
//                         0 => Direction::U,
//                         1 => Direction::L,
//                         2 => Direction::D,
//                         3 => Direction::R,
//                         _ => unreachable!(),
//                     };
//                     steps += 1;
//                     let r = game.step(dir);
//                     // println!("Step: {}", steps);
//                     // println!("Direction: {:?}", dir);
//                     // make_board_printing::<8>(&pos, &food, &bar).pretty_print();
//                     if r.is_ok() {
//                         if r.unwrap() {
//                             assert!(game.get_pair_vec().1.is_empty());
//                             success_count += 1;
//                             println!("Game over. Step: {}", steps);
//                             break;
//                         }
//                     } else {
//                         println!("Error: {:?}", r);
//                         break;
//                     }
//                     if steps >= 200 {
//                         println!("Too many steps, manually check");
//                         println!(
//                             "board: {:?}",
//                             make_board_printing::<8>(&pos, &food, &bar).pretty_print()
//                         );
//                         println!("Step: {}", steps);
//                         break;
//                     }
//                 }
//             }
//         }
//         println!("Success count: {}", success_count);
//         println!("Unreachable count: {}", unreachable_count);
//     }
// }
