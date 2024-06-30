/*
 * Modifications Copyright (C) 2023 Intel Corporation
 *
 * This Program is subject to the terms of the GNU Lesser General Public License v3.0 or later
 * 
 * If a copy of the license was not distributed with this file, you can obtain one at 
 * https://www.gnu.org/licenses/lgpl-3.0-standalone.html
 * 
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */


#include "hip/hip_runtime.h"
//
// Created by amr-nasr on 11/21/19.
//

#include <operations/components/independents/concrete/computation-kernels/isotropic/SecondOrderComputationKernel.hpp>

#include <timer/Timer.h>
#include <memory-manager/MemoryManager.h>

#include <cstring>
#include <cassert>

#include "Logging.h"

////#define fma(a, b, c) (a) * (b) + (c)

#define HALF_LENGTH_CUDA 8
//////using namespace cl::sycl;
using namespace std;
using namespace operations::components;
using namespace operations::dataunits;
using namespace operations::common;

template void SecondOrderComputationKernel::Compute<true, O_2>();

template void SecondOrderComputationKernel::Compute<true, O_4>();

template void SecondOrderComputationKernel::Compute<true, O_8>();

template void SecondOrderComputationKernel::Compute<true, O_12>();

template void SecondOrderComputationKernel::Compute<true, O_16>();

template void SecondOrderComputationKernel::Compute<false, O_2>();

template void SecondOrderComputationKernel::Compute<false, O_4>();

template void SecondOrderComputationKernel::Compute<false, O_8>();

template void SecondOrderComputationKernel::Compute<false, O_12>();

template void SecondOrderComputationKernel::Compute<false, O_16>();

#ifdef RESTRICT
__global__ void ComputeKernel(float * __restrict__ curr_base,
                              float * __restrict__ prev_base,
                              float * __restrict__ next_base,
                              float * __restrict__ vel_base,
                              int const block_z,
                              float * __restrict__ mpCoeffX,
                              float * __restrict__ mpCoeffZ,
                              float const mCoeffXYZ,
                              int   * __restrict__ mpVerticalIdx,
                              size_t const nx

        )
{
#else
__global__ void ComputeKernel(float * curr_base,
                              float * prev_base,
                              float * next_base,
                              float * vel_base,
                              int const block_z,
                              float * mpCoeffX,
                              float * mpCoeffZ,
                              float const mCoeffXYZ,
                              int   * mpVerticalIdx,
                              size_t const nx

        )
{
#endif

    int const hl = HALF_LENGTH_CUDA; 
    const float *current = curr_base;
    const float *prev = prev_base;
    float *next = next_base;
    const float *vel = vel_base;
    const float *c_x = mpCoeffX; 
    const float *c_z = mpCoeffZ; 
    const float c_xyz = mCoeffXYZ;
    const int *v = mpVerticalIdx; 
    const int idx_range = block_z; 
    const int pad = 0;

    __shared__ float local[136]; // shared memory 
    int iIndexX = blockDim.x * blockIdx.x + threadIdx.x;
    int iIndexY = blockDim.y * blockIdx.y + threadIdx.y;


    int idx = iIndexX + hl + (iIndexY * idx_range + hl) * nx;
    size_t id0 = threadIdx.x;

    size_t identifiant = (id0 + hl);
    float c_x_loc[HALF_LENGTH_CUDA];
    float c_z_loc[HALF_LENGTH_CUDA];
    int v_end = v[HALF_LENGTH_CUDA - 1];
    float front[HALF_LENGTH_CUDA + 1];
    float back[HALF_LENGTH_CUDA];


    bool copyHaloX = false;

    //============================================
    const unsigned int items_X = blockDim.x;
    float modulo_var_a, modulo_var_b;
    //============================================

    if (id0 < HALF_LENGTH_CUDA) {
        copyHaloX = true;
        //============================================
        modulo_var_a =  current[idx - HALF_LENGTH_CUDA];
        modulo_var_b =  current[idx + items_X];
        //============================================
    }


    for (unsigned int iter = 0; iter <= HALF_LENGTH_CUDA; iter++) {
        // ld-use 16/30
        front[iter] = current[idx + nx * iter];
    }
    for (unsigned int iter = 1; iter <= HALF_LENGTH_CUDA; iter++) {
        // ld-use 11/14
        back[iter - 1] = current[idx - nx * iter];
        c_x_loc[iter - 1] = c_x[iter - 1];
        c_z_loc[iter - 1] = c_z[iter - 1];
    }

    for (int i = 0; i < idx_range; i++) {

        local[identifiant] = front[0];
        if (copyHaloX) {
            // TODO ld-use 1/2
            local[identifiant - HALF_LENGTH_CUDA] = modulo_var_a;
            local[identifiant + items_X] = modulo_var_b;
        }
        // TODO wait
        __syncthreads();

        //============================================
        if (copyHaloX && i+1 < idx_range) {
            modulo_var_a =  current[idx + nx - HALF_LENGTH_CUDA];
            modulo_var_b =  current[idx + nx + items_X];
        }
        //============================================

        ////it.barrier(access::fence_space::local_space);
        float value = 0;
        value = fmaf(local[identifiant], c_xyz, value);
        for (int iter = 1; iter <= HALF_LENGTH_CUDA; iter++) {
            value = fmaf(local[identifiant - iter], c_x_loc[iter - 1], value);
            value = fmaf(local[identifiant + iter], c_x_loc[iter - 1], value);
        }
        for (int iter = 1; iter <= HALF_LENGTH_CUDA; iter++) {
            value = fmaf(front[iter], c_z_loc[iter - 1], value);
            value = fmaf(back[iter - 1], c_z_loc[iter - 1], value);
        }
        // TODO ld-use 43 / 44
        // vel[idx]
        // prev[idx]
        value = fmaf(vel[idx], value, -prev[idx]);
        value = fmaf(2.0f, local[identifiant], value);

        // TODO Write
        // next[idx]
        next[idx] = value;
        idx += nx;
        for (unsigned int iter = HALF_LENGTH_CUDA - 1; iter > 0; iter--) {
            back[iter] = back[iter - 1];
        }
        back[0] = front[0];
        for (unsigned int iter = 0; iter < HALF_LENGTH_CUDA; iter++) {
            front[iter] = front[iter + 1];
        }
        // Only one new data-point read from global memory
        // in z-dimension (depth)

        // TODO ld-use 10
        // current[idx + v_end]
        front[HALF_LENGTH_CUDA] = current[idx + v_end];
    }
}


template<bool IS_2D_, HALF_LENGTH HALF_LENGTH_>
void SecondOrderComputationKernel::Compute() {
    // Read parameters into local variables to be shared.

    size_t nx = mpGridBox->GetActualWindowSize(X_AXIS);
    size_t nz = mpGridBox->GetActualWindowSize(Z_AXIS);

    float *prev_base = mpGridBox->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetNativePointer();
    float *curr_base = mpGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();
    float *next_base = mpGridBox->Get(WAVE | GB_PRSS | NEXT | DIR_Z)->GetNativePointer();

    float *vel_base = mpGridBox->Get(PARM | WIND | GB_VEL)->GetNativePointer();

    // Pre-compute the coefficients for each direction.
    int hl = HALF_LENGTH_;

    int compute_nz = mpGridBox->GetComputationGridSize(Z_AXIS) / mpParameters->GetBlockZ();
    assert(mpGridBox->GetComputationGridSize(X_AXIS) % mpParameters->GetBlockX() == 0);
    dim3 const cuBlockSize(mpParameters->GetBlockX(), 1), cuGridSize(mpGridBox->GetComputationGridSize(X_AXIS) / mpParameters->GetBlockX(), compute_nz);
    /////std::cout << "Grid Size : " << cuGridSize.x  << ", " << cuGridSize.y << std::endl;
    /////std::cout << "Block Size: " << cuBlockSize.x << ", " << cuBlockSize.y << std::endl;
 
    printf("%d iterations\n", mpParameters->GetBlockZ());
    hipLaunchKernelGGL(ComputeKernel, cuGridSize, cuBlockSize, 0, 0,
                  curr_base,
                  prev_base,
                  next_base,
                  vel_base,
                  mpParameters->GetBlockZ(),
                  mpCoeffX->GetNativePointer(),
                  mpCoeffZ->GetNativePointer(), 
                  mCoeffXYZ,
                  mpVerticalIdx->GetNativePointer(),
                  nx);
    checkLastHIPError();
}
