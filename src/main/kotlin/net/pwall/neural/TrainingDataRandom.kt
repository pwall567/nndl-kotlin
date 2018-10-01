/*
 * @(#) TrainingDataRandom.java
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

package net.pwall.neural

import java.util.Random

class TrainingDataRandom(private val source: TrainingDataSource) : TrainingDataSource {

    private val index = Array(source.getSize()) { i -> i }

    fun randomise(r: Random) {
        var i = getSize()
        while (i > 1) {
            val x = r.nextInt(i)
            val previous = index[x]
            System.arraycopy(index, x + 1, index, x, i - x - 1)
            index[--i] = previous
        }
    }

    override fun getItem(index: Int) = source.getItem(this.index[index])

    override fun getSize(): Int = index.size

}
