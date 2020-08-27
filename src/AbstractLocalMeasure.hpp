#pragma once

#include "SpatialModel.hpp"
#include "PlatoStaticsTypes.hpp"
#include "SimplexMechanics.hpp"
#include <Teuchos_ParameterList.hpp>

namespace Plato
{

/******************************************************************************//**
 * \brief Abstract local measure class for use in Augmented Lagrange constraint formulation
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType, typename SimplexPhysics>
class AbstractLocalMeasure
{
protected:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */
    static constexpr Plato::OrdinalType mNumVoigtTerms = SimplexPhysics::mNumVoigtTerms; /*!< number of Voigt terms */
    static constexpr Plato::OrdinalType mNumNodesPerCell = SimplexPhysics::mNumNodesPerCell; /*!< number of nodes per cell/element */

    using StateT = typename EvaluationType::StateScalarType; /*!< state variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using ResultT = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */

    const Plato::SpatialDomain & mSpatialDomain;

    const std::string mName; /*!< Local measure name */

public:
    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aInputParams input parameters database
     * \param [in] aName local measure name
     **********************************************************************************/
    AbstractLocalMeasure(
        const Plato::SpatialDomain   & aSpatialDomain,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aName
    ) :
        mSpatialDomain (aSpatialDomain),
        mName          (aName)
    {
    }

    /******************************************************************************//**
     * \brief Constructor tailored for unit testing
     * \param [in] aName local measure name
     **********************************************************************************/
// TODO this constructor prohibits mSpatialDomain from being a const reference member
// since const references must be set with initializer list
/*
    AbstractLocalMeasure(const std::string & aName) : mName(aName)
    {
    }
*/

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    virtual ~AbstractLocalMeasure()
    {
    }

    /******************************************************************************//**
     * \brief Evaluate local measure
     * \param [in] aState 2D container of state variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell criterion values
    **********************************************************************************/
    virtual void
    operator()(
        const Plato::ScalarMultiVectorT <StateT>  & aStateWS,
        const Plato::ScalarArray3DT     <ConfigT> & aConfigWS,
              Plato::ScalarVectorT      <ResultT> & aResultWS ) = 0;

    /******************************************************************************//**
     * \brief Get local measure name
     * \return Return local measure name
     **********************************************************************************/
    virtual std::string getName()
    {
        return mName;
    }
};
//class AbstractLocalMeasure

}
//namespace Plato
