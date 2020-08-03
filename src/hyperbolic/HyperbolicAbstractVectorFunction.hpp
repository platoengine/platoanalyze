#ifndef ABSTRACT_VECTOR_FUNCTION_HYPERBOLIC_HPP
#define ABSTRACT_VECTOR_FUNCTION_HYPERBOLIC_HPP

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Hyperbolic
{

/******************************************************************************/
template<typename EvaluationType>
class AbstractVectorFunction
/******************************************************************************/
{
protected:
    Omega_h::Mesh& mMesh;
    Plato::DataMap& mDataMap;
    Omega_h::MeshSets& mMeshSets;
    std::vector<std::string> mDofNames;
    std::vector<std::string> mDofDotNames;

public:
    /******************************************************************************/
    explicit 
    AbstractVectorFunction(
        Omega_h::Mesh            & aMesh, 
        Omega_h::MeshSets        & aMeshSets,
        Plato::DataMap           & aDataMap,
        std::vector<std::string>   aStateNames,
        std::vector<std::string>   aStateDotNames
    ) :
    /******************************************************************************/
            mMesh(aMesh),
            mDataMap(aDataMap),
            mMeshSets(aMeshSets),
            mDofNames(aStateNames),
            mDofDotNames(aStateDotNames)
    {
    }
    /******************************************************************************/
    virtual ~AbstractVectorFunction()
    /******************************************************************************/
    {
    }

    /****************************************************************************//**
    * \brief Return reference to Omega_h mesh data base 
    ********************************************************************************/
    decltype(mMesh) getMesh() const
    {
        return (mMesh);
    }

    /****************************************************************************//**
    * \brief Return reference to Omega_h mesh sets 
    ********************************************************************************/
    decltype(mMeshSets) getMeshSets() const
    {
        return (mMeshSets);
    }

    /****************************************************************************//**
    * \brief Return reference to state index map
    ********************************************************************************/
    decltype(mDofNames) getDofNames() const
    {
        return (mDofNames);
    }

    /****************************************************************************//**
    * \brief Return reference to state dot index map
    ********************************************************************************/
    decltype(mDofDotNames) getDofDotNames() const
    {
        return (mDofDotNames);
    }


    /******************************************************************************/
    virtual void
    evaluate(const Plato::ScalarMultiVectorT< typename EvaluationType::StateScalarType       > & aState,
             const Plato::ScalarMultiVectorT< typename EvaluationType::StateDotScalarType    > & aStateDot,
             const Plato::ScalarMultiVectorT< typename EvaluationType::StateDotDotScalarType > & aStateDotDot,
             const Plato::ScalarMultiVectorT< typename EvaluationType::ControlScalarType     > & aControl,
             const Plato::ScalarArray3DT    < typename EvaluationType::ConfigScalarType      > & aConfig,
                   Plato::ScalarMultiVectorT< typename EvaluationType::ResultScalarType      > & aResult,
             Plato::Scalar aTimeStep = 0.0) const = 0;
    /******************************************************************************/
};

} // namespace Hyperbolic

} // namespace Plato

#endif
