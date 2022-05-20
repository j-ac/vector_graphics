#![allow(dead_code)]
use std::ops;
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone)]
pub struct Matrix<const ROWS: usize, const COLS: usize> {
    matrix: [[f64; COLS]; ROWS],
}

//Matrix with one column
type ColumnVector<const SIZE: usize> = Matrix<SIZE, 1>;

impl<const ROWS: usize, const COLS: usize> Index<(usize, usize)> for Matrix<ROWS, COLS> {
    type Output = f64;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.matrix[index.0][index.1]
    }
}

impl<const ROWS: usize, const COLS: usize> IndexMut<(usize, usize)> for Matrix<ROWS, COLS> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.matrix[index.0][index.1]
    }
}

impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS> {
    pub fn zero() -> Self {
        Self {
            matrix: [[0.0; COLS]; ROWS],
        }
    }

    pub fn ones() -> Self {
        Self {
            matrix: [[1.0; COLS]; ROWS],
        }
    }
}

//For methods requiring square matrices
impl<const ROWS: usize> Matrix<ROWS, ROWS> {
    pub fn identity() -> Self {
        let mut ret = Self::zero();
        for i in 0..ROWS {
            ret[(i, i)] = 1.0;
        }

        ret
    }
}

//Matrix addition with + operator
impl<const ROWS: usize, const COLS: usize> ops::Add<Matrix<ROWS, COLS>> for Matrix<ROWS, COLS> {
    type Output = Matrix<ROWS, COLS>;
    fn add(self, rhs: Self::Output) -> Self::Output {
        let mut ret = Self::zero();

        for ((i, lhs_row), rhs_row) in self.matrix.iter().enumerate().zip(rhs.matrix.iter()) {
            for ((j, lhs_val), rhs_val) in lhs_row.iter().enumerate().zip(rhs_row.iter()) {
                ret[(i, j)] = lhs_val + rhs_val;
            }
        }

        ret
    }
}

//Vector Addition with += operator
impl<const ROWS: usize, const COLS: usize> ops::AddAssign<Matrix<ROWS, COLS>>
    for Matrix<ROWS, COLS>
{
    fn add_assign(&mut self, rhs: Self) {
        //TODO
        todo!();
    }
}

//Vector-Scalar multiplication with * operator
impl<const ROWS: usize, const COLS: usize> ops::Mul<f64> for Matrix<ROWS, COLS> {
    type Output = Matrix<ROWS, COLS>;
    fn mul(self, rhs: f64) -> Self::Output {
        for row in self.matrix.iter() {
            row.map(|x| x * rhs);
        }

        self
    }
}

#[derive(Debug)]
pub struct Coordinate3d {
    pos: ColumnVector<3>,
    edges: Vec<Coordinate3d>,
}

//Translate a coordinate by a vector
impl Coordinate3d {
    fn translate(&mut self, vector: ColumnVector<3>) {
        self.pos += vector;
    }
}

#[derive(Debug)]
pub struct Mesh {
    vertices: Vec<Coordinate3d>,
    origin: Coordinate3d,
}

impl Mesh {
    pub fn translate(&mut self, vector: ColumnVector<3>) {
        for vertex in self.vertices.iter_mut() {
            vertex.translate(vector.clone()); //TODO: avoid clone() by implementing + by reference.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_ones() {
        assert_eq!(
            Matrix::<3, 4>::ones().matrix,
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0]
            ]
        );
    }

    #[test]
    fn constructor_zero() {
        assert_eq!(
            Matrix::<3, 4>::zero().matrix,
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0]
            ]
        );
    }

    #[test]
    fn constructor_identity() {
        assert_eq!(
            Matrix::<4, 4>::identity().matrix,
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        );
    }

    #[test]
    fn matrix_index() {
        let mtx: Matrix<3, 3> = Matrix::<3, 3>::identity();
        assert_eq!(mtx[(0, 0)], 1.0);
        assert_eq!(mtx[(2, 1)], 0.0);
    }

    #[test]
    fn matrix_add() {
        assert_eq!(
            (Matrix::<2, 2>::ones() + Matrix::<2, 2>::ones()).matrix,
            [[2.0, 2.0], [2.0, 2.0]]
        );
    }
}
