/*
 * PlatoTypes.hpp
 *
 *  Created on: Jul 12, 2018
 */

#ifndef SRC_PLATO_PLATOTYPES_HPP_
#define SRC_PLATO_PLATOTYPES_HPP_

#include <Kokkos_Core.hpp>

namespace Plato
{

using Scalar = double;
using OrdinalType = int;
using ExecSpace = Kokkos::DefaultExecutionSpace;
// #ifdef KOKKOS_ENABLE_CUDA_UVM
//   using MemSpace = typename Kokkos::CudaSpace;
// #else
  using MemSpace = typename ExecSpace::memory_space;
// #endif
using Layout = Kokkos::LayoutRight;

} // namespace Plato

#endif /* SRC_PLATO_PLATOTYPES_HPP_ */
