/*
 * @(#) InputDataSource.kt
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

import net.pwall.neural.TrainingData
import net.pwall.neural.TrainingDataSource
import net.pwall.neural.test.images.MNISTImageData
import net.pwall.neural.test.images.MNISTLabelData

class InputDataSource(val imageData: MNISTImageData, val labelData: MNISTLabelData) : TrainingDataSource {

    val pixels = imageData.numRows * imageData.numCols

    override fun getItem(index: Int) = InputData(index)

    override fun getSize() = imageData.numImages

    inner class InputData(private val image: Int) : TrainingData {

        override fun getInputs() = DoubleArray(pixels) { i -> imageData.getPixelValue(image, i).toDouble() / 256 }

        override fun getOutputs() : DoubleArray {
            val n = labelData.getLabelValue(image)
            return DoubleArray(10) { i -> if (i == n) 1.0 else 0.0 }
        }

        override fun getHighestOutputIndex() = labelData.getLabelValue(image)

    }

}
