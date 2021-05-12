/*
 * PlatoStaticsTypes.hpp
 *
 *  Created on: Jul 12, 2018
 */

#ifndef SRC_PLATO_PLATOSTATICSTYPES_HPP_
#define SRC_PLATO_PLATOSTATICSTYPES_HPP_

#include <map>
#include <vector>

#include "alg/CrsMatrix.hpp"
#include "AnalyzeMacros.hpp"
#include "PlatoTypes.hpp"

namespace Plato
{

using CrsMatrixType      = typename Plato::CrsMatrix<Plato::OrdinalType>;
using LocalOrdinalVector = typename Kokkos::View<Plato::OrdinalType*, Plato::MemSpace>;

template <typename ScalarType>
using ScalarVectorT = typename Kokkos::View<ScalarType*, Plato::MemSpace>;
using ScalarVector  = ScalarVectorT<Plato::Scalar>;

template <typename ScalarType>
using HostScalarVectorT = typename Kokkos::View<ScalarType*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
using HostScalarVector  = HostScalarVectorT<Plato::Scalar>;

template <typename ScalarType>
using ScalarMultiVectorT = typename Kokkos::View<ScalarType**, Plato::Layout, Plato::MemSpace>;
using ScalarMultiVector  = ScalarMultiVectorT<Plato::Scalar>;

template <typename ScalarType>
using HostMultiScalarVectorT = typename Kokkos::View<ScalarType**, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
using HostMultiScalarVector  = HostMultiScalarVectorT<Plato::Scalar>;

template <typename ScalarType>
using ScalarArray3DT = typename Kokkos::View<ScalarType***, Plato::Layout, Plato::MemSpace>;
using ScalarArray3D  = ScalarArray3DT<Plato::Scalar>;

template <typename ScalarType>
using HostScalarArray3DT = typename Kokkos::View<ScalarType***, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
using HostScalarArray3D  = HostScalarArray3DT<Plato::Scalar>;

struct DataMap
{
  std::map<std::string, Plato::Scalar> mScalarValues;
  std::map<std::string, Plato::ScalarVector> scalarVectors;
  std::map<std::string, Plato::ScalarMultiVector> scalarMultiVectors;
  std::map<std::string, Plato::ScalarArray3D> scalarArray3Ds;

  std::map<std::string, Plato::ScalarVectorT<Plato::OrdinalType>> ordinalVectors;
  std::map<std::string, Plato::ScalarVector> scalarNodeFields;
  std::map<std::string, Plato::ScalarVector> vectorNodeFields;

  std::vector<DataMap> stateDataMaps;

  void clearAll()
  {
    clearStates();
    mScalarValues.clear();
    scalarVectors.clear();
    scalarMultiVectors.clear();
    scalarArray3Ds.clear();
    scalarNodeFields.clear();
    vectorNodeFields.clear();
  }

  void clearStates()
  {
    stateDataMaps.clear();
  }

  void saveState()
  {
    stateDataMaps.push_back(getState());

    mScalarValues.clear();
    scalarVectors.clear();
    scalarMultiVectors.clear();
    scalarArray3Ds.clear();

    scalarNodeFields.clear();
    vectorNodeFields.clear();
  }

  DataMap getState() const
  {
    DataMap tState(*this);
    tState.stateDataMaps.clear();
    return tState;
  }

  DataMap getState(int aStateIndex) const
  {
    if ( stateDataMaps.size() == 0 )
    {
      return getState();
    }
    else 
    {
      auto tNumStates = stateDataMaps.size();
      if ( aStateIndex < 0 || aStateIndex >= tNumStates )
      {
        THROWERR("Requested a state that doesn't exist");
      }
      else
      {
        return stateDataMaps[aStateIndex];
      }
    }
  }
};
// struct DataMap

} // namespace Plato

#endif /* SRC_PLATO_PLATOSTATICSTYPES_HPP_ */
