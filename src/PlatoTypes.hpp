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
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
  using OrdinalType = long long int;
#else
  using OrdinalType = int;
#endif
using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace = typename ExecSpace::memory_space;
using DeviceType = Kokkos::Device<ExecSpace, MemSpace>;

// using Layout = typename ExecSpace::array_layout;
using Layout = typename Kokkos::LayoutRight;

} // namespace Plato

#endif /* SRC_PLATO_PLATOTYPES_HPP_ */
