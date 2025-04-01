#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    U = 0,
    L = 1,
    D = 2,
    R = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tile {
    Empty(usize),
    SnakeHead,
    SnakeBody,
    Food(usize),
    Barrier,
}

impl Direction {
    pub fn to_delta(self) -> (i32, i32) {
        match self {
            Direction::U => (0, 1),
            Direction::L => (-1, 0),
            Direction::D => (0, -1),
            Direction::R => (1, 0),
        }
    }
}

#[derive(Clone)]
pub struct DirectionMarker(usize);

impl DirectionMarker {
    const U: usize = 1;
    const L: usize = 2;
    const D: usize = 4;
    const R: usize = 5;

    pub fn empty() -> Self {
        DirectionMarker(0)
    }

    pub fn set(&mut self, dir: Direction) {
        match dir {
            Direction::U => self.0 |= Self::U,
            Direction::L => self.0 |= Self::L,
            Direction::D => self.0 |= Self::D,
            Direction::R => self.0 |= Self::R,
        }
    }

    pub fn contains(&self, dir: Direction) -> bool {
        match dir {
            Direction::U => self.0 & Self::U != 0,
            Direction::L => self.0 & Self::L != 0,
            Direction::D => self.0 & Self::D != 0,
            Direction::R => self.0 & Self::R != 0,
        }
    }

    pub fn from_dir(dir: Direction) -> Self {
        match dir {
            Direction::U => DirectionMarker(Self::U),
            Direction::L => DirectionMarker(Self::L),
            Direction::D => DirectionMarker(Self::D),
            Direction::R => DirectionMarker(Self::R),
        }
    }
    
    pub fn intersects(&self, other: &Self) -> bool {
        self.0 & other.0 != 0
    }

    pub fn union(&self, other: &Self) -> Self {
        DirectionMarker(self.0 | other.0)
    }

    pub fn devide(&self, other: &Self) -> Self {
        DirectionMarker(self.0 & !other.0)
    }

    pub fn count(&self) -> usize {
        self.0.count_ones() as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Point {
    x: i32,
    y: i32,
}

impl Point {
    pub fn new(x: i32, y: i32) -> Self {
        Point { x, y }
    }

    pub fn step(&self, direction: Direction) -> Self {
        let (dx, dy) = direction.to_delta();
        Point::new(self.x + dx, self.y + dy)
    }

    pub fn is_adjacent(&self, other: &Self) -> bool {
        (self.x - other.x).abs() + (self.y - other.y).abs() == 1
    }

    pub fn inbounds(&self, n: usize) -> bool {
        self.x > 0 && self.x <= n as i32 && self.y > 0 && self.y <= n as i32
    }

    pub fn make_pair(&self) -> (i32, i32) {
        (self.x, self.y)
    }
}

pub trait AsPoint {
    fn as_point(&self) -> Point;
}

impl AsPoint for (i32, i32) {
    fn as_point(&self) -> Point {
        Point::new(self.0, self.1)
    }
}

impl AsPoint for Point {
    fn as_point(&self) -> Point {
        *self
    }
}

impl AsPoint for &[i32] {
    fn as_point(&self) -> Point {
        assert_eq!(self.len(), 2);
        Point::new(self[0], self[1])
    }
}

pub struct Board<const N: usize> {
    tiles: [[Tile; N]; N],
}

impl<const N: usize> Board<N> {
    pub fn new() -> Self {
        Board {
            tiles: [[Tile::Empty(usize::MAX); N]; N],
        }
    }

    pub fn inbounds(&self, point: impl AsPoint) -> bool {
        let point = point.as_point();
        point.x > 0 && point.x <= N as i32 && point.y > 0 && point.y <= N as i32
    }

    pub fn set_tile(&mut self, point: impl AsPoint, tile: Tile) {
        let point = point.as_point();
        if self.inbounds(point) {
            self.tiles[(point.x - 1) as usize][(point.y - 1) as usize] = tile;
        }
    }

    pub fn get_tile(&self, point: impl AsPoint) -> Option<Tile> {
        let point = point.as_point();
        if self.inbounds(point) {
            Some(self.tiles[(point.x - 1) as usize][(point.y - 1) as usize])
        } else {
            None
        }
    }

    #[allow(dead_code)]
    pub fn pretty_print(&self) {
        // for row in self.tiles.iter().rev() {
        //     for tile in row.iter() {
        //         match tile {
        //             Tile::Empty(_) => print!(". "),
        //             Tile::SnakeHead => print!("H "),
        //             Tile::SnakeBody => print!("B "),
        //             Tile::Food(_) => print!("F "),
        //             Tile::Barrier => print!("# "),
        //         }
        //     }
        //     println!();
        // }
        for i in (1..=N).rev() {
            for j in 1..=N {
                let point = Point::new(j as i32, i as i32);
                match self.get_tile(point) {
                    Some(Tile::Empty(_)) => print!(". "),
                    Some(Tile::SnakeHead) => print!("H "),
                    Some(Tile::SnakeBody) => print!("B "),
                    Some(Tile::Food(_)) => print!("F "),
                    Some(Tile::Barrier) => print!("# "),
                    None => print!("? "),
                }
            }
        println!();
        }
    }
}
