#ifndef PLATO_CELL_FORCING_HPP
#define PLATO_CELL_FORCING_HPP

#include <Omega_h_matrix.hpp>

#include "PlatoStaticsTypes.hpp"

/******************************************************************************/
/*! Add forcing for homogenization cell problem.

 given a view, subtract the forcing column.
 */
/******************************************************************************/
namespace Plato
{

template<Plato::OrdinalType NumTerms>
class CellForcing
{
private:

    const Omega_h::Matrix<NumTerms, NumTerms> mCellStiffness;
    const Plato::OrdinalType mColumnIndex;

public:

    CellForcing(const Omega_h::Matrix<NumTerms, NumTerms> aCellStiffness, Plato::OrdinalType aColumnIndex) :
            mCellStiffness(aCellStiffness),
            mColumnIndex(aColumnIndex)
    {
    }

    template<typename TensorScalarType>
    void add(Kokkos::View<TensorScalarType**, Plato::Layout, Plato::MemSpace> const& aTensor) const
    {
        Plato::OrdinalType tNumCells = aTensor.extent(0);
        Plato::OrdinalType tNumTerms = aTensor.extent(1);

        // A lambda inside a member function captures the "this"
        // pointer not the actual members as such a local copy of the
        // data is need here for the lambda to capture everything.

        // If compiling with C++17 (Clang as the compiler or CUDA 11
        // with Kokkos 3.2). And using KOKKOS_CLASS_LAMBDA instead of
        // KOKKOS_EXPRESSION. Then the memeber data can be used
        // directly.
        auto tCellStiffness = mCellStiffness;
        auto tColumnIndex = mColumnIndex;

        Kokkos::parallel_for(Kokkos::RangePolicy < Plato::OrdinalType > (0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
        {
            // add forcing
            //
            for(Plato::OrdinalType tTermIndex = 0; tTermIndex < tNumTerms; tTermIndex++)
            {
                aTensor(aCellOrdinal,tTermIndex) -= tCellStiffness(tTermIndex, tColumnIndex);
            }
        }, "Add Forcing");
    }
};

} // end namespace Plato
#endif
