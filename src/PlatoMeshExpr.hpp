#pragma once

#include <Omega_h_expr.hpp>
#include <Omega_h_mesh.hpp>

#include "SpatialModel.hpp"
#include "OmegaHUtilities.hpp"
#include "ImplicitFunctors.hpp"
#include "Plato_TopOptFunctors.hpp"

namespace Plato
{

/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
void
getFunctionValues(
          Plato::ScalarArray3D   aQuadraturePoints,
    const std::string          & aFuncString,
          Omega_h::Reals       & aFxnValues
)
/******************************************************************************/
{
    Plato::OrdinalType numCells = aQuadraturePoints.extent(0);
    Plato::OrdinalType numPoints = aQuadraturePoints.extent(1);

    auto x_coords = Plato::create_omega_h_write_array<Plato::Scalar>("forcing function x coords", numCells * numPoints);
    auto y_coords = Plato::create_omega_h_write_array<Plato::Scalar>("forcing function y coords", numCells * numPoints);
    auto z_coords = Plato::create_omega_h_write_array<Plato::Scalar>("forcing function z coords", numCells * numPoints);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, numCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
    {
        Plato::OrdinalType entryOffset = aCellOrdinal * numPoints;
        for (Plato::OrdinalType ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
        {
            if (SpaceDim > 0) x_coords[entryOffset+ptOrdinal] = aQuadraturePoints(aCellOrdinal,ptOrdinal,0);
            if (SpaceDim > 1) y_coords[entryOffset+ptOrdinal] = aQuadraturePoints(aCellOrdinal,ptOrdinal,1);
            if (SpaceDim > 2) z_coords[entryOffset+ptOrdinal] = aQuadraturePoints(aCellOrdinal,ptOrdinal,2);
        }
    },
                         "fill coords");

    Omega_h::ExprReader reader(numCells * numPoints, SpaceDim);
    if(SpaceDim > 0)
        reader.register_variable("x", Omega_h::any(Omega_h::Reals(x_coords)));
    if(SpaceDim > 1)
        reader.register_variable("y", Omega_h::any(Omega_h::Reals(y_coords)));
    if(SpaceDim > 2)
        reader.register_variable("z", Omega_h::any(Omega_h::Reals(z_coords)));

    auto result = reader.read_string(aFuncString, "Integrand");
    reader.repeat(result);
    aFxnValues = Omega_h::any_cast<Omega_h::Reals>(result);
}

/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
void
mapPoints(
    const Plato::SpatialDomain     & aSpatialDomain,
          Plato::ScalarMultiVector   aRefPoints,
          Plato::ScalarArray3D       aMappedPoints
)
/******************************************************************************/
{
    Plato::OrdinalType tNumCells = aSpatialDomain.numCells();
    Plato::OrdinalType tNumPoints = aMappedPoints.extent(1);

    Kokkos::deep_copy(aMappedPoints, Plato::Scalar(0.0)); // initialize to 0

    Plato::NodeCoordinate<SpaceDim> tNodeCoordinate(&(aSpatialDomain.Mesh));

    auto tCellOrdinals = aSpatialDomain.cellOrdinals();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
    {
        auto tCellOrdinal = tCellOrdinals[aCellOrdinal];
        for (Plato::OrdinalType ptOrdinal=0; ptOrdinal<tNumPoints; ptOrdinal++)
        {
            Plato::OrdinalType tNodeOrdinal;
            Scalar tFinalNodeValue = 1.0;
            for (tNodeOrdinal=0; tNodeOrdinal<SpaceDim; tNodeOrdinal++)
            {
                Scalar tNodeValue = aRefPoints(ptOrdinal,tNodeOrdinal);
                tFinalNodeValue -= tNodeValue;
                for (Plato::OrdinalType d=0; d<SpaceDim; d++)
                {
                    aMappedPoints(aCellOrdinal, ptOrdinal, d) += tNodeValue * tNodeCoordinate(tCellOrdinal, tNodeOrdinal, d);
                }
            }
            tNodeOrdinal = SpaceDim;
            for (Plato::OrdinalType d=0; d<SpaceDim; d++)
            {
                aMappedPoints(aCellOrdinal, ptOrdinal, d) += tFinalNodeValue * tNodeCoordinate(tCellOrdinal, tNodeOrdinal, d);
            }
        }
    });
}

/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
void
mapPoints(
    const Plato::SpatialModel      & aSpatialModel,
          Plato::ScalarMultiVector   aRefPoints,
          Plato::ScalarArray3D       aMappedPoints
)
/******************************************************************************/
{
    Plato::OrdinalType tNumCells = aSpatialModel.Mesh.nelems();
    Plato::OrdinalType tNumPoints = aMappedPoints.extent(1);

    Kokkos::deep_copy(aMappedPoints, Plato::Scalar(0.0)); // initialize to 0

    Plato::NodeCoordinate<SpaceDim> tNodeCoordinate(&(aSpatialModel.Mesh));

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
    {
        for (Plato::OrdinalType ptOrdinal=0; ptOrdinal<tNumPoints; ptOrdinal++)
        {
            Plato::OrdinalType tNodeOrdinal;
            Scalar tFinalNodeValue = 1.0;
            for (tNodeOrdinal=0; tNodeOrdinal<SpaceDim; tNodeOrdinal++)
            {
                Scalar tNodeValue = aRefPoints(ptOrdinal,tNodeOrdinal);
                tFinalNodeValue -= tNodeValue;
                for (Plato::OrdinalType d=0; d<SpaceDim; d++)
                {
                    aMappedPoints(aCellOrdinal, ptOrdinal, d) += tNodeValue * tNodeCoordinate(aCellOrdinal, tNodeOrdinal, d);
                }
            }
            tNodeOrdinal = SpaceDim;
            for (Plato::OrdinalType d=0; d<SpaceDim; d++)
            {
                aMappedPoints(aCellOrdinal, ptOrdinal, d) += tFinalNodeValue * tNodeCoordinate(aCellOrdinal, tNodeOrdinal, d);
            }
        }
    });
}

}
