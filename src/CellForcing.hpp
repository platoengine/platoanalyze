#ifndef LGR_PLATO_CELL_FORCING_HPP
#define LGR_PLATO_CELL_FORCING_HPP

#include <Omega_h_matrix.hpp>

#include "plato/PlatoStaticsTypes.hpp"

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
    void add(Kokkos::View<TensorScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& aTensor) const
    {
        Plato::OrdinalType tNumCells = aTensor.extent(0);
        Plato::OrdinalType tNumTerms = aTensor.extent(1);
        auto tCellStiffness = mCellStiffness;
        auto& tTensor = aTensor;
        auto tColumnIndex = mColumnIndex;
        Kokkos::parallel_for(Kokkos::RangePolicy < Plato::OrdinalType > (0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
        {
            // add forcing
            //
            for(Plato::OrdinalType tTermIndex = 0; tTermIndex < tNumTerms; tTermIndex++)
            {
                tTensor(aCellOrdinal,tTermIndex) -= tCellStiffness(tTermIndex, tColumnIndex);
            }
        }, "Add Forcing");
    }
};

} // end namespace Plato
#endif
