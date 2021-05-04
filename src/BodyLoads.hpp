#ifndef BODYLOADS_HPP
#define BODYLOADS_HPP

#include <Omega_h_expr.hpp>
#include <Omega_h_mesh.hpp>

#include <Teuchos_ParameterList.hpp>

#include "alg/Basis.hpp"
#include "UtilsOmegaH.hpp"
#include "alg/Cubature.hpp"
#include "PlatoTypes.hpp"
#include "SpatialModel.hpp"
#include "ImplicitFunctors.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "PlatoMeshExpr.hpp"

namespace Plato
{

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
