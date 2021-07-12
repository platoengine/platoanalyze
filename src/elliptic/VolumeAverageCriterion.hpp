#pragma once

#include <memory>
#include <cassert>
#include <vector>

#include <Omega_h_mesh.hpp>

//#include "BLAS1.hpp"
#include "PlatoStaticsTypes.hpp"
#include "WorksetBase.hpp"
#include "elliptic/ScalarFunctionBaseFactory.hpp"
#include "elliptic/PhysicsScalarFunction.hpp"
#include "elliptic/DivisionFunction.hpp"
#include "AnalyzeMacros.hpp"
#include <Omega_h_expr.hpp>
#include "alg/Cubature.hpp"
#include "PlatoMeshExpr.hpp"
#include "UtilsOmegaH.hpp"

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Electromechanics.hpp"
#include "Thermomechanics.hpp"

#ifdef PLATO_STABILIZED
#include "StabilizedMechanics.hpp"
#include "StabilizedThermomechanics.hpp"
#endif

#include <Teuchos_ParameterList.hpp>

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Volume average criterion class
 **********************************************************************************/
template<typename PhysicsT>
class VolumeAverageCriterion : public Plato::Elliptic::ScalarFunctionBase, public Plato::WorksetBase<PhysicsT>
{
private:
    static constexpr Plato::OrdinalType mSpaceDim = PhysicsT::mNumSpatialDims; /*!< spatial dimensions */

    using Residual  = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using GradientU = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Jacobian;
    using GradientX = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientX;
    using GradientZ = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientZ;

    std::shared_ptr<Plato::Elliptic::DivisionFunction<PhysicsT>> mDivisionFunction;

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap& mDataMap; /*!< PLATO Engine and Analyze data map */

    std::string mFunctionName; /*!< User defined function name */

    std::string mSpatialWeightingFunctionString = "1.0"; /*!< Spatial weighting function string of x, y, z coordinates  */

    //std::map<std::string, Plato::Scalar> mMaterialDensities; /*!< material density */

    /******************************************************************************//**
     * \brief Initialization of Mass Properties Function
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void
    initialize(
        Teuchos::ParameterList & aInputParams
    )
    {
        auto params = aInputParams.sublist("Criteria").get<Teuchos::ParameterList>(mFunctionName);
        if (params.isType<std::string>("Function"))
            mSpatialWeightingFunctionString = params.get<std::string>("Function");

        createDivisionFunction(mSpatialModel, aInputParams);
    }


    /******************************************************************************//**
     * \brief Create the volume function only
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aInputParams parameter list
     * \return physics scalar function
    **********************************************************************************/
    std::shared_ptr<Plato::Elliptic::PhysicsScalarFunction<PhysicsT>>
    getVolumeFunction(
        const Plato::SpatialModel & aSpatialModel,
        Teuchos::ParameterList & aInputParams
    )
    {
        std::shared_ptr<Plato::Elliptic::PhysicsScalarFunction<PhysicsT>> tVolumeFunction =
             std::make_shared<Plato::Elliptic::PhysicsScalarFunction<PhysicsT>>(aSpatialModel, mDataMap);
        tVolumeFunction->setFunctionName("Volume Function");

        typename PhysicsT::FunctionFactory tFactory;
        std::string tFunctionType = "volume average criterion denominator";

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            Plato::ScalarVector tSpatialWeights = this->computeSpatialWeightingValues(tDomain);

            std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<Residual>> tValue = 
                 tFactory.template createScalarFunction<Residual>(tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);
            tValue->setSpatialWeights(tSpatialWeights);
            tVolumeFunction->setEvaluator(tValue, tName);

            std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<GradientU>> tGradientU = 
                 tFactory.template createScalarFunction<GradientU>(tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);
            tGradientU->setSpatialWeights(tSpatialWeights);
            tVolumeFunction->setEvaluator(tGradientU, tName);

            std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<GradientZ>> tGradientZ = 
                 tFactory.template createScalarFunction<GradientZ>(tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);
            tGradientZ->setSpatialWeights(tSpatialWeights);
            tVolumeFunction->setEvaluator(tGradientZ, tName);

            std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<GradientX>> tGradientX = 
                 tFactory.template createScalarFunction<GradientX>(tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);
            tGradientX->setSpatialWeights(tSpatialWeights);
            tVolumeFunction->setEvaluator(tGradientX, tName);
        }
        return tVolumeFunction;
    }

    /******************************************************************************//**
     * \brief Create the division function
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aInputParams parameter list
    **********************************************************************************/
    void
    createDivisionFunction(
        const Plato::SpatialModel & aSpatialModel,
        Teuchos::ParameterList & aInputParams
    )
    {
        const std::string tNumeratorName = "Volume Average Criterion Numerator";
        std::shared_ptr<Plato::Elliptic::PhysicsScalarFunction<PhysicsT>> tNumerator =
             std::make_shared<Plato::Elliptic::PhysicsScalarFunction<PhysicsT>>(aSpatialModel, mDataMap);
        tNumerator->setFunctionName(tNumeratorName);

        typename PhysicsT::FunctionFactory tFactory;
        std::string tFunctionType = "volume average criterion numerator";

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            Plato::ScalarVector tSpatialWeights = this->computeSpatialWeightingValues(tDomain);

            std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<Residual>> tNumeratorValue = 
                 tFactory.template createScalarFunction<Residual>(tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);
            tNumeratorValue->setSpatialWeights(tSpatialWeights);
            tNumerator->setEvaluator(tNumeratorValue, tName);

            std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<GradientU>> tNumeratorGradientU = 
                 tFactory.template createScalarFunction<GradientU>(tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);
            tNumeratorGradientU->setSpatialWeights(tSpatialWeights);
            tNumerator->setEvaluator(tNumeratorGradientU, tName);

            std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<GradientZ>> tNumeratorGradientZ = 
                 tFactory.template createScalarFunction<GradientZ>(tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);
            tNumeratorGradientZ->setSpatialWeights(tSpatialWeights);
            tNumerator->setEvaluator(tNumeratorGradientZ, tName);

            std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<GradientX>> tNumeratorGradientX = 
                 tFactory.template createScalarFunction<GradientX>(tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);
            tNumeratorGradientX->setSpatialWeights(tSpatialWeights);
            tNumerator->setEvaluator(tNumeratorGradientX, tName);
        }

        const std::string tDenominatorName = "Volume Function";
        std::shared_ptr<Plato::Elliptic::PhysicsScalarFunction<PhysicsT>> tDenominator = 
             getVolumeFunction(aSpatialModel, aInputParams);
        tDenominator->setFunctionName(tDenominatorName);

        mDivisionFunction =
             std::make_shared<Plato::Elliptic::DivisionFunction<PhysicsT>>(aSpatialModel, mDataMap);
        mDivisionFunction->allocateNumeratorFunction(tNumerator);
        mDivisionFunction->allocateDenominatorFunction(tDenominator);
        mDivisionFunction->setFunctionName("Volume Average Criterion Division Function");
    }


public:
    /******************************************************************************//**
     * \brief Primary volume average criterion constructor
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aInputParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    VolumeAverageCriterion(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
              std::string            & aName
    ) :
        Plato::WorksetBase<PhysicsT>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap),
        mFunctionName (aName)
    {
        initialize(aInputParams);
    }


    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aState 1D view of state variables
     * \param [in] aControl 1D view of control variables
     **********************************************************************************/
    void
    updateProblem(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aControl
    ) const override
    {
        mDivisionFunction->updateProblem(aState, aControl);
    }

    /******************************************************************************//**
     * \brief Compute values of the spatial weighting function
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \return scalar vector containing the spatial weights for the specified domain
    **********************************************************************************/
    Plato::ScalarVector computeSpatialWeightingValues(const Plato::SpatialDomain & aSpatialDomain)
    {
      // get refCellQuadraturePoints, quadratureWeights
      //
      Plato::OrdinalType tQuadratureDegree = 1;

      Plato::OrdinalType tNumPoints = Plato::Cubature::getNumCubaturePoints(mSpaceDim, tQuadratureDegree);

      Plato::ScalarMultiVector tRefCellQuadraturePoints("ref quadrature points", tNumPoints, mSpaceDim);
      Plato::ScalarVector      tQuadratureWeights("quadrature weights", tNumPoints);

      Plato::Cubature::getCubature(mSpaceDim, tQuadratureDegree, tRefCellQuadraturePoints, tQuadratureWeights);

      // get basis values
      //
      Plato::Basis tBasis(mSpaceDim);
      Plato::OrdinalType tNumFields = tBasis.basisCardinality();
      Plato::ScalarMultiVector tRefCellBasisValues("ref basis values", tNumFields, tNumPoints);
      tBasis.getValues(tRefCellQuadraturePoints, tRefCellBasisValues);

      // map points to physical space
      //
      Plato::OrdinalType tNumCells = aSpatialDomain.numCells();
      Plato::ScalarArray3D tQuadraturePoints("quadrature points", tNumCells, tNumPoints, mSpaceDim);

      Plato::mapPoints<mSpaceDim>(aSpatialDomain, tRefCellQuadraturePoints, tQuadraturePoints);

      // get integrand values at quadrature points
      //
      Omega_h::Reals tFxnValues;
      Plato::getFunctionValues<mSpaceDim>(tQuadraturePoints, mSpatialWeightingFunctionString, tFxnValues);

      // Copy the result into a ScalarVector
      Plato::ScalarVector tSpatialWeightingValues("spatial weights", tFxnValues.size());
      Plato::OrdinalType tNumLocalVals = tFxnValues.size();
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumLocalVals), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
      {
          tSpatialWeightingValues(aOrdinal) = tFxnValues[aOrdinal];
      }, "copy vector");

      return tSpatialWeightingValues;
    }

    /******************************************************************************//**
     * \brief Evaluate Mass Properties Function
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return scalar function evaluation
    **********************************************************************************/
    Plato::Scalar
    value(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const override
    {
        Plato::Scalar tFunctionValue = mDivisionFunction->value(aSolution, aControl, aTimeStep);
        return tFunctionValue;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the Mass Properties Function with respect to (wrt) the state variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_u(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::OrdinalType    aStepIndex,
              Plato::Scalar         aTimeStep = 0.0
    ) const override
    {
        Plato::ScalarVector tGradientU = mDivisionFunction->gradient_u(aSolution, aControl, aStepIndex, aTimeStep);
        return tGradientU;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the Mass Properties Function with respect to (wrt) the configuration
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_x(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const override
    {
        Plato::ScalarVector tGradientX = mDivisionFunction->gradient_x(aSolution, aControl, aTimeStep);
        return tGradientX;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the Mass Properties Function with respect to (wrt) the control
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_z(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const override
    {
        Plato::ScalarVector tGradientZ = mDivisionFunction->gradient_z(aSolution, aControl, aTimeStep);
        return tGradientZ;
    }


    /******************************************************************************//**
     * \brief Return user defined function name
     * \return User defined function name
    **********************************************************************************/
    std::string name() const
    {
        return mFunctionName;
    }
};
// class VolumeAverageCriterion

} // namespace Elliptic

} // namespace Plato


#ifdef PLATOANALYZE_2D
extern template class Plato::Elliptic::VolumeAverageCriterion<::Plato::Thermal<2>>;
extern template class Plato::Elliptic::VolumeAverageCriterion<::Plato::Mechanics<2>>;
extern template class Plato::Elliptic::VolumeAverageCriterion<::Plato::Electromechanics<2>>;
extern template class Plato::Elliptic::VolumeAverageCriterion<::Plato::Thermomechanics<2>>;
#ifdef PLATO_STABILIZED
extern template class Plato::Elliptic::VolumeAverageCriterion<::Plato::StabilizedMechanics<2>>;
extern template class Plato::Elliptic::VolumeAverageCriterion<::Plato::StabilizedThermomechanics<2>>;
#endif
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::Elliptic::VolumeAverageCriterion<::Plato::Thermal<3>>;
extern template class Plato::Elliptic::VolumeAverageCriterion<::Plato::Mechanics<3>>;
extern template class Plato::Elliptic::VolumeAverageCriterion<::Plato::Electromechanics<3>>;
extern template class Plato::Elliptic::VolumeAverageCriterion<::Plato::Thermomechanics<3>>;
#ifdef PLATO_STABILIZED
extern template class Plato::Elliptic::VolumeAverageCriterion<::Plato::StabilizedMechanics<3>>;
extern template class Plato::Elliptic::VolumeAverageCriterion<::Plato::StabilizedThermomechanics<3>>;
#endif
#endif
