/*
 * @(#) TestNDArray.kt
 *
 * nndl-kotlin Neural Networks and Deep Learning - Kotlin
 * Copyright (c) 2018 Peter Wall
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package net.pwall.neural.test

import net.pwall.neural.NDArray
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class TestNDArray {

    private val delta = 1e-10

    @Test
    fun testConstructor() { // implicitly tests get(x, y) / [x, y] as well
        val nda1 = NDArray(3, 4)
        assertEquals(3, nda1.dim1)
        assertEquals(4, nda1.dim2)
        for (i in 0..2)
            for (j in 0..3)
                assertEquals(0.0, nda1[i, j], delta)
    }

    @Test
    fun testConstructorWithArray() {
        val nda1 = NDArray(3, 4, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0))
        assertEquals(3, nda1.dim1)
        assertEquals(4, nda1.dim2)
        assertEquals(1.0, nda1[0, 0], delta)
        assertEquals(2.0, nda1[0, 1], delta)
        assertEquals(3.0, nda1[0, 2], delta)
        assertEquals(4.0, nda1[0, 3], delta)
        assertEquals(5.0, nda1[1, 0], delta)
        assertEquals(6.0, nda1[1, 1], delta)
        assertEquals(12.0, nda1[2, 3], delta)
    }

    @Test
    fun testConstructorError() {
        var nda1 : NDArray? = null
        val throwable = assertThrows(IllegalArgumentException::class.java) {
            nda1 = NDArray(2, 2, doubleArrayOf(1.0, 2.0, 3.0))
        }
        assertNull(nda1)
        assertEquals("Array is incorrect size", throwable.message)
    }

    @Test
    fun testSet() {
        val nda1 = NDArray(3, 4)
        nda1[0, 0] = 1.0
        nda1[0, 1] = 2.0
        nda1[0, 2] = 3.0
        nda1[0, 3] = 4.0
        nda1[1, 0] = 5.0
        nda1[1, 1] = 6.0
        nda1[2, 3] = 12.0
        assertEquals(1.0, nda1[0, 0], delta)
        assertEquals(2.0, nda1[0, 1], delta)
        assertEquals(3.0, nda1[0, 2], delta)
        assertEquals(4.0, nda1[0, 3], delta)
        assertEquals(5.0, nda1[1, 0], delta)
        assertEquals(6.0, nda1[1, 1], delta)
        assertEquals(0.0, nda1[1, 2], delta)
        assertEquals(12.0, nda1[2, 3], delta)
    }

    @Test
    fun testPlus() {
        val nda1 = NDArray(2, 3, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        val nda2 = NDArray(2, 3, doubleArrayOf(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        val nda3 = nda1 + nda2
        assertEquals(2, nda3.dim1)
        assertEquals(3, nda3.dim2)
        assertEquals(1.1, nda3[0, 0], delta)
        assertEquals(2.2, nda3[0, 1], delta)
        assertEquals(3.3, nda3[0, 2], delta)
        assertEquals(4.4, nda3[1, 0], delta)
        assertEquals(5.5, nda3[1, 1], delta)
        assertEquals(6.6, nda3[1, 2], delta)
        // now check that the original objects were not modified
        assertEquals(1.0, nda1[0, 0], delta)
        assertEquals(2.0, nda1[0, 1], delta)
        assertEquals(3.0, nda1[0, 2], delta)
        assertEquals(4.0, nda1[1, 0], delta)
        assertEquals(5.0, nda1[1, 1], delta)
        assertEquals(6.0, nda1[1, 2], delta)
        assertEquals(0.1, nda2[0, 0], delta)
        assertEquals(0.2, nda2[0, 1], delta)
        assertEquals(0.3, nda2[0, 2], delta)
        assertEquals(0.4, nda2[1, 0], delta)
        assertEquals(0.5, nda2[1, 1], delta)
        assertEquals(0.6, nda2[1, 2], delta)
    }

    @Test
    fun testPlusError() {
        val nda1 = NDArray(2, 3, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        val nda2 = NDArray(3, 2, doubleArrayOf(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        var nda3 : NDArray? = null
        val throwable = assertThrows(IllegalArgumentException::class.java) {
            nda3 = nda1 + nda2
        }
        assertNull(nda3)
        assertEquals("Arrays must be same dimensions", throwable.message)
    }

    @Test
    fun testPlusAssign() {
        val nda1 = NDArray(2, 3, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        val nda2 = NDArray(2, 3, doubleArrayOf(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        nda1 += nda2
        assertEquals(2, nda1.dim1)
        assertEquals(3, nda1.dim2)
        assertEquals(1.1, nda1[0, 0], delta)
        assertEquals(2.2, nda1[0, 1], delta)
        assertEquals(3.3, nda1[0, 2], delta)
        assertEquals(4.4, nda1[1, 0], delta)
        assertEquals(5.5, nda1[1, 1], delta)
        assertEquals(6.6, nda1[1, 2], delta)
        // now check that the other object was not modified
        assertEquals(0.1, nda2[0, 0], delta)
        assertEquals(0.2, nda2[0, 1], delta)
        assertEquals(0.3, nda2[0, 2], delta)
        assertEquals(0.4, nda2[1, 0], delta)
        assertEquals(0.5, nda2[1, 1], delta)
        assertEquals(0.6, nda2[1, 2], delta)
    }

    @Test
    fun testPlusAssignError() {
        val nda1 = NDArray(2, 3, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        val nda2 = NDArray(3, 2, doubleArrayOf(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        val throwable = assertThrows(IllegalArgumentException::class.java) {
            nda1 += nda2
        }
        assertEquals("Arrays must be same dimensions", throwable.message)
    }

    @Test
    fun testMinus() {
        val nda1 = NDArray(2, 3, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        val nda2 = NDArray(2, 3, doubleArrayOf(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        val nda3 = nda1 - nda2
        assertEquals(2, nda3.dim1)
        assertEquals(3, nda3.dim2)
        assertEquals(0.9, nda3[0, 0], delta)
        assertEquals(1.8, nda3[0, 1], delta)
        assertEquals(2.7, nda3[0, 2], delta)
        assertEquals(3.6, nda3[1, 0], delta)
        assertEquals(4.5, nda3[1, 1], delta)
        assertEquals(5.4, nda3[1, 2], delta)
        // now check that the original objects were not modified
        assertEquals(1.0, nda1[0, 0], delta)
        assertEquals(2.0, nda1[0, 1], delta)
        assertEquals(3.0, nda1[0, 2], delta)
        assertEquals(4.0, nda1[1, 0], delta)
        assertEquals(5.0, nda1[1, 1], delta)
        assertEquals(6.0, nda1[1, 2], delta)
        assertEquals(0.1, nda2[0, 0], delta)
        assertEquals(0.2, nda2[0, 1], delta)
        assertEquals(0.3, nda2[0, 2], delta)
        assertEquals(0.4, nda2[1, 0], delta)
        assertEquals(0.5, nda2[1, 1], delta)
        assertEquals(0.6, nda2[1, 2], delta)
    }

    @Test
    fun testMinusError() {
        val nda1 = NDArray(2, 3, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        val nda2 = NDArray(3, 2, doubleArrayOf(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        var nda3 : NDArray? = null
        val throwable = assertThrows(IllegalArgumentException::class.java) {
            nda3 = nda1 - nda2
        }
        assertNull(nda3)
        assertEquals("Arrays must be same dimensions", throwable.message)
    }

    @Test
    fun testMinusAssign() {
        val nda1 = NDArray(2, 3, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        val nda2 = NDArray(2, 3, doubleArrayOf(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        nda1 -= nda2
        assertEquals(2, nda1.dim1)
        assertEquals(3, nda1.dim2)
        assertEquals(0.9, nda1[0, 0], delta)
        assertEquals(1.8, nda1[0, 1], delta)
        assertEquals(2.7, nda1[0, 2], delta)
        assertEquals(3.6, nda1[1, 0], delta)
        assertEquals(4.5, nda1[1, 1], delta)
        assertEquals(5.4, nda1[1, 2], delta)
        // now check that the other object was not modified
        assertEquals(0.1, nda2[0, 0], delta)
        assertEquals(0.2, nda2[0, 1], delta)
        assertEquals(0.3, nda2[0, 2], delta)
        assertEquals(0.4, nda2[1, 0], delta)
        assertEquals(0.5, nda2[1, 1], delta)
        assertEquals(0.6, nda2[1, 2], delta)
    }

    @Test
    fun testMinusAssignError() {
        val nda1 = NDArray(2, 3, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        val nda2 = NDArray(3, 2, doubleArrayOf(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        val throwable = assertThrows(IllegalArgumentException::class.java) {
            nda1 -= nda2
        }
        assertEquals("Arrays must be same dimensions", throwable.message)
    }

    @Test
    fun testTimes() {
        val nda1 = NDArray(2, 3, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        val nda2 = NDArray(2, 3, doubleArrayOf(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        val nda3 = nda1 * nda2
        assertEquals(2, nda3.dim1)
        assertEquals(3, nda3.dim2)
        assertEquals(0.1, nda3[0, 0], delta)
        assertEquals(0.4, nda3[0, 1], delta)
        assertEquals(0.9, nda3[0, 2], delta)
        assertEquals(1.6, nda3[1, 0], delta)
        assertEquals(2.5, nda3[1, 1], delta)
        assertEquals(3.6, nda3[1, 2], delta)
        // now check that the original objects were not modified
        assertEquals(1.0, nda1[0, 0], delta)
        assertEquals(2.0, nda1[0, 1], delta)
        assertEquals(3.0, nda1[0, 2], delta)
        assertEquals(4.0, nda1[1, 0], delta)
        assertEquals(5.0, nda1[1, 1], delta)
        assertEquals(6.0, nda1[1, 2], delta)
        assertEquals(0.1, nda2[0, 0], delta)
        assertEquals(0.2, nda2[0, 1], delta)
        assertEquals(0.3, nda2[0, 2], delta)
        assertEquals(0.4, nda2[1, 0], delta)
        assertEquals(0.5, nda2[1, 1], delta)
        assertEquals(0.6, nda2[1, 2], delta)
    }

    @Test
    fun testTimesAssign() {
        val nda1 = NDArray(2, 3, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        val nda2 = NDArray(2, 3, doubleArrayOf(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        nda1 *= nda2
        assertEquals(2, nda1.dim1)
        assertEquals(3, nda1.dim2)
        assertEquals(0.1, nda1[0, 0], delta)
        assertEquals(0.4, nda1[0, 1], delta)
        assertEquals(0.9, nda1[0, 2], delta)
        assertEquals(1.6, nda1[1, 0], delta)
        assertEquals(2.5, nda1[1, 1], delta)
        assertEquals(3.6, nda1[1, 2], delta)
        // now check that the other object was not modified
        assertEquals(0.1, nda2[0, 0], delta)
        assertEquals(0.2, nda2[0, 1], delta)
        assertEquals(0.3, nda2[0, 2], delta)
        assertEquals(0.4, nda2[1, 0], delta)
        assertEquals(0.5, nda2[1, 1], delta)
        assertEquals(0.6, nda2[1, 2], delta)
    }

    @Test
    fun testDot() {
        val nda1 = NDArray(4, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0))
        val nda2 = NDArray(2, 3, doubleArrayOf(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        val nda3 = nda1 dot nda2
        assertEquals(4, nda3.dim1)
        assertEquals(3, nda3.dim2)
        assertEquals(0.9, nda3[0, 0], delta)
        assertEquals(1.2, nda3[0, 1], delta)
        assertEquals(1.5, nda3[0, 2], delta)
        assertEquals(1.9, nda3[1, 0], delta)
        assertEquals(2.6, nda3[1, 1], delta)
        assertEquals(3.3, nda3[1, 2], delta)
        assertEquals(2.9, nda3[2, 0], delta)
        assertEquals(4.0, nda3[2, 1], delta)
        assertEquals(5.1, nda3[2, 2], delta)
        assertEquals(3.9, nda3[3, 0], delta)
        assertEquals(5.4, nda3[3, 1], delta)
        assertEquals(6.9, nda3[3, 2], delta)
        // now check that the original objects were not modified
        assertEquals(1.0, nda1[0, 0], delta)
        assertEquals(2.0, nda1[0, 1], delta)
        assertEquals(3.0, nda1[1, 0], delta)
        assertEquals(4.0, nda1[1, 1], delta)
        assertEquals(5.0, nda1[2, 0], delta)
        assertEquals(6.0, nda1[2, 1], delta)
        assertEquals(7.0, nda1[3, 0], delta)
        assertEquals(8.0, nda1[3, 1], delta)
        assertEquals(0.1, nda2[0, 0], delta)
        assertEquals(0.2, nda2[0, 1], delta)
        assertEquals(0.3, nda2[0, 2], delta)
        assertEquals(0.4, nda2[1, 0], delta)
        assertEquals(0.5, nda2[1, 1], delta)
        assertEquals(0.6, nda2[1, 2], delta)
    }

    @Test
    fun testDotError() {
        val nda1 = NDArray(4, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0))
        val nda2 = NDArray(3, 2, doubleArrayOf(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        var nda3 : NDArray? = null
        val throwable = assertThrows(IllegalArgumentException::class.java) {
            nda3 = nda1 dot nda2
        }
        assertNull(nda3)
        assertEquals("Array dimensions not compatible", throwable.message)
    }

}
