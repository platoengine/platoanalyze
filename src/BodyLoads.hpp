#ifndef BODYLOADS_HPP
#define BODYLOADS_HPP

#include <Omega_h_expr.hpp>
#include <Omega_h_mesh.hpp>

#include <Teuchos_ParameterList.hpp>

#include "alg/Basis.hpp"
#include "alg/Cubature.hpp"
#include "SpatialModel.hpp"
#include "OmegaHUtilities.hpp"
#include "ImplicitFunctors.hpp"
#include "Plato_TopOptFunctors.hpp"

//#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
void getFunctionValues(Kokkos::View<Plato::Scalar***, Plato::Layout, Plato::MemSpace> aQuadraturePoints,
                       const std::string& aFuncString,
                       Omega_h::Reals& aFxnValues)
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
/*!
 \brief Class for essential boundary conditions.
 */
template<typename EvaluationType, typename PhysicsType>
class BodyLoad
/******************************************************************************/
{
private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */
    static constexpr Plato::OrdinalType mNumDofsPerNode = PhysicsType::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr Plato::OrdinalType mNumNodesPerCell = PhysicsType::mNumNodesPerCell; /*!< number of nodes per cell/element */

protected:
    const std::string mName;
    const Plato::OrdinalType mDof;
    const std::string mFuncString;

public:

    /**************************************************************************/
    BodyLoad<EvaluationType, PhysicsType>(const std::string &aName, Teuchos::ParameterList &aParam) :
            mName(aName),
            mDof(aParam.get<Plato::OrdinalType>("Index", 0)),
            mFuncString(aParam.get<std::string>("Function"))
    {
    }
    /**************************************************************************/

    ~BodyLoad()
    {
    }

    /**************************************************************************/
    template<typename StateScalarType, typename ControlScalarType, typename ResultScalarType>
    void
    get(
        const Plato::SpatialDomain                         & aSpatialDomain,
        const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarMultiVectorT<ResultScalarType>  & aResult,
              Plato::Scalar                                  aScale
    ) const
    {
        // get refCellQuadraturePoints, quadratureWeights
        //
        Plato::OrdinalType tQuadratureDegree = 1;

        Plato::OrdinalType tNumPoints = Plato::Cubature::getNumCubaturePoints(mSpaceDim, tQuadratureDegree);

        Kokkos::View<Plato::Scalar**, Plato::Layout, Plato::MemSpace>
            tRefCellQuadraturePoints("ref quadrature points", tNumPoints, mSpaceDim);
        Kokkos::View<Plato::Scalar*, Plato::Layout, Plato::MemSpace> tQuadratureWeights("quadrature weights", tNumPoints);

        Plato::Cubature::getCubature(mSpaceDim, tQuadratureDegree, tRefCellQuadraturePoints, tQuadratureWeights);

        // get basis values
        //
        Plato::Basis tBasis(mSpaceDim);
        Plato::OrdinalType tNumFields = tBasis.basisCardinality();
        Kokkos::View<Plato::Scalar**, Plato::Layout, Plato::MemSpace>
            tRefCellBasisValues("ref basis values", tNumFields, tNumPoints);
        tBasis.getValues(tRefCellQuadraturePoints, tRefCellBasisValues);

        // map points to physical space
        //
        Plato::OrdinalType tNumCells = aSpatialDomain.numCells();
        Kokkos::View<Plato::Scalar***, Plato::Layout, Plato::MemSpace>
            tQuadraturePoints("quadrature points", tNumCells, tNumPoints, mSpaceDim);

        Plato::mapPoints<mSpaceDim>(aSpatialDomain, tRefCellQuadraturePoints, tQuadraturePoints);

        // get integrand values at quadrature points
        //
        Omega_h::Reals tFxnValues;
        Plato:: getFunctionValues<mSpaceDim>(tQuadraturePoints, mFuncString, tFxnValues);

        // integrate and assemble
        //
        auto tDof = mDof;
        auto tCellOrdinals = aSpatialDomain.cellOrdinals();
        Plato::JacobianDet<mSpaceDim> tJacobianDet(&(aSpatialDomain.Mesh));
        Plato::VectorEntryOrdinal<mSpaceDim, mSpaceDim> tVectorEntryOrdinal(&(aSpatialDomain.Mesh));
        Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType &aCellOrdinal)
        {
            auto tCellOrdinal = tCellOrdinals[aCellOrdinal];
            auto tDetElemJacobian = fabs(tJacobianDet(tCellOrdinal));
            ControlScalarType tDensity = Plato::cell_density<mNumNodesPerCell>(aCellOrdinal, aControl);

            auto tEntryOffset = aCellOrdinal * tNumPoints;
            for (Plato::OrdinalType tPtOrdinal=0; tPtOrdinal < tNumPoints; tPtOrdinal++)
            {
                auto tFxnValue = tFxnValues[tEntryOffset + tPtOrdinal];
                auto tWeight = aScale * tQuadratureWeights(tPtOrdinal) * tDetElemJacobian;
                for (Plato::OrdinalType tFieldOrdinal = 0; tFieldOrdinal < tNumFields; tFieldOrdinal++)
                {

                    aResult(aCellOrdinal,tFieldOrdinal*mNumDofsPerNode+tDof) +=
                            tWeight * tFxnValue * tRefCellBasisValues(tFieldOrdinal,tPtOrdinal) * tDensity;
                }
            }
        }, "assemble RHS");
    }

};
// end class BodyLoad

/******************************************************************************/
/*!
 \brief Owner class that contains a vector of BodyLoad objects.
 */
template<typename EvaluationType, typename PhysicsType>
class BodyLoads
/******************************************************************************/
{
private:
    std::vector<std::shared_ptr<BodyLoad<EvaluationType, PhysicsType>>> mBodyLoads;

public:

    /******************************************************************************//**
     * \brief Constructor that parses and creates a vector of BodyLoad objects based on
     *   the ParameterList.
     * \param aParams Body Loads sublist with input parameters
    **********************************************************************************/
    BodyLoads(Teuchos::ParameterList &aParams) :
            mBodyLoads()
    {
        for(Teuchos::ParameterList::ConstIterator tIndex = aParams.begin(); tIndex != aParams.end(); ++tIndex)
        {
            const Teuchos::ParameterEntry & tEntry = aParams.entry(tIndex);
            const std::string &tName = aParams.name(tIndex);

            if(!tEntry.isList())
            {
                THROWERR("Parameter in Body Loads block not valid.  Expect lists only.");
            }

            Teuchos::ParameterList& tSublist = aParams.sublist(tName);
            std::shared_ptr<Plato::BodyLoad<EvaluationType, PhysicsType>> tBodyLoad;
            auto tNewBodyLoad = new Plato::BodyLoad<EvaluationType, PhysicsType>(tName, tSublist);
            tBodyLoad.reset(tNewBodyLoad);
            mBodyLoads.push_back(tBodyLoad);
        }
    }

    /**************************************************************************/
    /*!
     \brief Add the body load to the result workset
     */
    template<typename StateScalarType, typename ControlScalarType, typename ResultScalarType>
    void
    get(
        const Plato::SpatialDomain                         & aSpatialDomain,
              Plato::ScalarMultiVectorT<StateScalarType>     aState,
              Plato::ScalarMultiVectorT<ControlScalarType>   aControl,
              Plato::ScalarMultiVectorT<ResultScalarType>    aResult,
              Plato::Scalar                                  aScale = 1.0
    ) const
    {
        for(const auto & tBodyLoad : mBodyLoads)
        {
            tBodyLoad->get(aSpatialDomain, aState, aControl, aResult, aScale);
        }
    }
};

}

#endif
