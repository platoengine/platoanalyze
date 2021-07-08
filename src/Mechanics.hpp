#ifndef PLATO_MECHANICS_HPP
#define PLATO_MECHANICS_HPP

#include <memory>

#include "parabolic/AbstractScalarFunction.hpp"

#include "elliptic/AbstractScalarFunction.hpp"
#include "elliptic/InternalElasticEnergy.hpp"
#include "elliptic/ElastostaticResidual.hpp"
#include "elliptic/EffectiveEnergy.hpp"
#include "elliptic/StressPNorm.hpp"
#include "elliptic/VolAvgStressPNormDenominator.hpp"
#include "elliptic/SurfaceArea.hpp"
#include "elliptic/Volume.hpp"

#include "Plato_AugLagStressCriterionQuadratic.hpp"
#include "Plato_AugLagStressCriterionGeneral.hpp"
#include "Plato_AugLagStressCriterion.hpp"
#include "SimplexMechanics.hpp"
#include "IntermediateDensityPenalty.hpp"
#include "AnalyzeMacros.hpp"

#include "AbstractLocalMeasure.hpp"
#include "VonMisesLocalMeasure.hpp"
#include "TensileEnergyDensityLocalMeasure.hpp"

#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"

namespace Plato
{

namespace MechanicsFactory
{

  /******************************************************************************//**
   * \brief Create a local measure for use in augmented lagrangian quadratic
   * \param [in] aProblemParams input parameters
   * \param [in] aFuncName scalar function name
  **********************************************************************************/
  template <typename EvaluationType>
  inline std::shared_ptr<Plato::AbstractLocalMeasure<EvaluationType,Plato::SimplexMechanics<EvaluationType::SpatialDim>>> 
  create_local_measure(
      const Plato::SpatialDomain   & aSpatialDomain,
            Teuchos::ParameterList & aProblemParams,
      const std::string            & aFuncName
  )
  {
      auto tFunctionSpecs = aProblemParams.sublist("Criteria").sublist(aFuncName);
      auto tLocalMeasure = tFunctionSpecs.get<std::string>("Local Measure", "VonMises");
      auto tLowerLocalMeasT = Plato::tolower(tLocalMeasure);
      if(tLowerLocalMeasT == "vonmises")
      {
          return std::make_shared<VonMisesLocalMeasure<EvaluationType, Plato::SimplexMechanics<EvaluationType::SpatialDim>>>
              (aSpatialDomain, aProblemParams, "VonMises");
      }
      else if(tLowerLocalMeasT == "tensileenergydensity")
      {
          return std::make_shared<TensileEnergyDensityLocalMeasure<EvaluationType, Plato::SimplexMechanics<EvaluationType::SpatialDim>>>
              (aSpatialDomain, aProblemParams, "TensileEnergyDensity");
      }
      else
      {
          THROWERR("Unknown 'Local Measure' specified in 'Plato Problem' ParameterList")
      }
  }

/******************************************************************************//**
 * \brief Create elastostatics residual equation
 * \param [in] aSpatialDomain Plato Analyze spatial domain
 * \param [in] aDataMap Plato Analyze physics-based database
 * \param [in] aProblemParams input parameters
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<EvaluationType>>
elastostatics_residual(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
          std::string              aFuncName
)
{
    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<EvaluationType>> tOutput;
    auto tPenaltyParams = aProblemParams.sublist(aFuncName).sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    auto tLowerPenaltyT = Plato::tolower(tPenaltyType);
    if(tLowerPenaltyT == "simp")
    {
        tOutput = std::make_shared<Plato::Elliptic::ElastostaticResidual<EvaluationType, Plato::MSIMP>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams);
    }
    else
    if(tLowerPenaltyT == "ramp")
    {
        tOutput = std::make_shared<Plato::Elliptic::ElastostaticResidual<EvaluationType, Plato::RAMP>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams);
    }
    else
    if(tLowerPenaltyT == "heaviside")
    {
        tOutput = std::make_shared<Plato::Elliptic::ElastostaticResidual<EvaluationType, Plato::Heaviside>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams);
    }
    else
    if(tLowerPenaltyT == "nopenalty")
    {
        tOutput = std::make_shared<Plato::Elliptic::ElastostaticResidual<EvaluationType, Plato::NoPenalty>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams);
    }
    return (tOutput);
}
// function elastostatics_residual

/******************************************************************************//**
 * \brief Create augmented Lagrangian stress constraint criterion tailored for linear problems
 * \param [in] aSpatialDomain Plato Analyze spatial domain
 * \param [in] aDataMap Plato Analyze physics-based database
 * \param [in] aProblemParams input parameters
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
stress_constraint_linear(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
    const std::string            & aFuncName
)
{
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>> tOutput;
    tOutput = std::make_shared< Plato::AugLagStressCriterion<EvaluationType> >
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
    return (tOutput);
}

/******************************************************************************//**
 * \brief Create augmented Lagrangian stress constraint criterion tailored for general problems
 * \param [in] aSpatialDomain Plato Analyze spatial domain
 * \param [in] aDataMap Plato Analyze physics-based database
 * \param [in] aProblemParams input parameters
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
stress_constraint_general(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
    const std::string            & aFuncName
)
{
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>> tOutput;
    tOutput = std::make_shared <Plato::AugLagStressCriterionGeneral<EvaluationType> >
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
    return (tOutput);
}


/******************************************************************************//**
 * \brief Create augmented Lagrangian local constraint criterion with quadratic constraint formulation
 * \param [in] aSpatialDomain Plato Analyze spatial domain
 * \param [in] aDataMap Plato Analyze physics-based database
 * \param [in] aProblemParams input parameters
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
stress_constraint_quadratic(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
    const std::string            & aFuncName
)
{
    auto EvalMeasure = Plato::MechanicsFactory::create_local_measure<EvaluationType>(aSpatialDomain, aProblemParams, aFuncName);
    using Residual = typename Plato::ResidualTypes<Plato::SimplexMechanics<EvaluationType::SpatialDim>>;
    auto PODMeasure = Plato::MechanicsFactory::create_local_measure<Residual>(aSpatialDomain, aProblemParams, aFuncName);

    using SimplexT = Plato::SimplexMechanics<EvaluationType::SpatialDim>;
    std::shared_ptr<Plato::AugLagStressCriterionQuadratic<EvaluationType,SimplexT>> tOutput;
    tOutput = std::make_shared< Plato::AugLagStressCriterionQuadratic<EvaluationType,SimplexT> >
        (aSpatialDomain, aDataMap, aProblemParams, aFuncName);

    tOutput->setLocalMeasure(EvalMeasure, PODMeasure);
    return (tOutput);
}


/******************************************************************************//**
 * \brief Create internal elastic energy criterion
 * \param [in] aSpatialDomain Plato Analyze spatial domain
 * \param [in] aDataMap Plato Analyze physics-based database
 * \param [in] aProblemParams input parameters
 * \param [in] aFuncName vector function name
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
internal_elastic_energy(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
          std::string            & aFuncName
)
{
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>> tOutput;
    auto tPenaltyParams = aProblemParams.sublist("Criteria").sublist(aFuncName).sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    auto tLowerPenaltyT = Plato::tolower(tPenaltyType);
    if(tLowerPenaltyT == "simp")
    {
        tOutput = std::make_shared<Plato::Elliptic::InternalElasticEnergy<EvaluationType, Plato::MSIMP>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tLowerPenaltyT == "ramp")
    {
        tOutput = std::make_shared<Plato::Elliptic::InternalElasticEnergy<EvaluationType, Plato::RAMP>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tLowerPenaltyT == "heaviside")
    {
        tOutput = std::make_shared<Plato::Elliptic::InternalElasticEnergy<EvaluationType, Plato::Heaviside>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tLowerPenaltyT == "nopenalty")
    {
        tOutput = std::make_shared<Plato::Elliptic::InternalElasticEnergy<EvaluationType, Plato::NoPenalty>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    return (tOutput);
}
// function internal_elastic_energy

/******************************************************************************//**
 * \brief Create stress p-norm criterion
 * \param [in] aSpatialDomain Plato Analyze spatial domain
 * \param [in] aDataMap Plato Analyze physics-based database
 * \param [in] aProblemParams input parameters
 * \param [in] aFuncName vector function name
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
stress_p_norm(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
          std::string            & aFuncName
)
{
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>> tOutput;
    auto tPenaltyParams = aProblemParams.sublist("Criteria").sublist(aFuncName).sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    auto tLowerPenaltyT = Plato::tolower(tPenaltyType);
    if(tLowerPenaltyT == "simp")
    {
        tOutput = std::make_shared<Plato::Elliptic::StressPNorm<EvaluationType, Plato::MSIMP>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tLowerPenaltyT == "ramp")
    {
        tOutput = std::make_shared<Plato::Elliptic::StressPNorm<EvaluationType, Plato::RAMP>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tLowerPenaltyT == "heaviside")
    {
        tOutput = std::make_shared<Plato::Elliptic::StressPNorm<EvaluationType, Plato::Heaviside>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tLowerPenaltyT == "nopenalty")
    {
        tOutput = std::make_shared<Plato::Elliptic::StressPNorm<EvaluationType, Plato::NoPenalty>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    return (tOutput);
}
// function stress_p_norm

/******************************************************************************//**
 * \brief Create volume average stress p-norm denominator
 * \param [in] aSpatialDomain Plato Analyze spatial domain
 * \param [in] aDataMap Plato Analyze physics-based database
 * \param [in] aProblemParams input parameters
 * \param [in] aFuncName vector function name
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
vol_avg_stress_p_norm_denominator(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
          std::string            & aFuncName
)
{
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>> tOutput;
    auto tPenaltyParams = aProblemParams.sublist("Criteria").sublist(aFuncName).sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    auto tLowerPenaltyT = Plato::tolower(tPenaltyType);
    if(tLowerPenaltyT == "simp")
    {
        tOutput = std::make_shared<Plato::Elliptic::VolAvgStressPNormDenominator<EvaluationType, Plato::MSIMP>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tLowerPenaltyT == "ramp")
    {
        tOutput = std::make_shared<Plato::Elliptic::VolAvgStressPNormDenominator<EvaluationType, Plato::RAMP>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tLowerPenaltyT == "heaviside")
    {
        tOutput = std::make_shared<Plato::Elliptic::VolAvgStressPNormDenominator<EvaluationType, Plato::Heaviside>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tLowerPenaltyT == "nopenalty")
    {
        tOutput = std::make_shared<Plato::Elliptic::VolAvgStressPNormDenominator<EvaluationType, Plato::NoPenalty>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    return (tOutput);
}
// function stress_p_norm

/******************************************************************************//**
 * \brief Create volume criterion
 * \param [in] aSpatialDomain Plato Analyze spatial domain
 * \param [in] aDataMap Plato Analyze physics-based database
 * \param [in] aProblemParams input parameters
 * \param [in] aFuncName scalar function name
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
volume_criterion(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
          std::string            & aFuncName
)
{
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>> tOutput;
    auto tPenaltyParams = aProblemParams.sublist("Criteria").sublist(aFuncName).sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    auto tLowerPenaltyT = Plato::tolower(tPenaltyType);
    if(tLowerPenaltyT == "simp")
    {
        tOutput = std::make_shared<Plato::Elliptic::Volume<EvaluationType, Plato::MSIMP>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tLowerPenaltyT == "ramp")
    {
        tOutput = std::make_shared<Plato::Elliptic::Volume<EvaluationType, Plato::RAMP>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tLowerPenaltyT == "heaviside")
    {
        tOutput = std::make_shared<Plato::Elliptic::Volume<EvaluationType, Plato::Heaviside>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tLowerPenaltyT == "nopenalty")
    {
        tOutput = std::make_shared<Plato::Elliptic::Volume<EvaluationType, Plato::NoPenalty>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    return (tOutput);
}
// function stress_p_norm

/******************************************************************************//**
 * \brief Create effective energy criterion
 * \param [in] aSpatialDomain Plato Analyze spatial domain
 * \param [in] aDataMap Plato Analyze physics-based database
 * \param [in] aProblemParams input parameters
 * \param [in] aFuncName vector function name
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
effective_energy(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
          std::string            & aFuncName
)
{
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>> tOutput;
    auto tPenaltyParams = aProblemParams.sublist("Criteria").sublist(aFuncName).sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    auto tLowerPenaltyT = Plato::tolower(tPenaltyType);
    if(tLowerPenaltyT == "simp")
    {
        tOutput = std::make_shared<Plato::Elliptic::EffectiveEnergy<EvaluationType, Plato::MSIMP>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tLowerPenaltyT == "ramp")
    {
        tOutput = std::make_shared<Plato::Elliptic::EffectiveEnergy<EvaluationType, Plato::RAMP>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tLowerPenaltyT == "heaviside")
    {
        tOutput = std::make_shared<Plato::Elliptic::EffectiveEnergy<EvaluationType, Plato::Heaviside>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tLowerPenaltyT == "nopenalty")
    {
        tOutput = std::make_shared<Plato::Elliptic::EffectiveEnergy<EvaluationType, Plato::NoPenalty>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    return (tOutput);
}
// function effective_energy

/******************************************************************************//**
 * \brief Factory for linear mechanics problem
 * \brief Create surface area scalar function
 * \param [in] aSpatialDomain Plato Analyze spatial domain
 * \param [in] aDataMap Plato Analyze physics-based database
 * \param [in] aProblemParams input parameters
 * \param [in] aFuncName vector function name
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
surface_area(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
    const std::string            & aFuncName
)
{
    return std::make_shared<Plato::Elliptic::SurfaceArea<EvaluationType>>(aSpatialDomain, aDataMap, aProblemParams, aFuncName);
}
// function surface area

/******************************************************************************//**
 * \brief Factory for linear mechanics problem
**********************************************************************************/
struct FunctionFactory
{
    /******************************************************************************//**
     * \brief Create a PLATO vector function (i.e. residual equation)
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap Plato Analyze physics-based database
     * \param [in] aProblemParams input parameters
     * \param [in] aPDE PDE type
    **********************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<EvaluationType>>
    createVectorFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams,
              std::string              aPDE)
    {
        auto tLowerPDE = Plato::tolower(aPDE);
        if(tLowerPDE == "elliptic")
        {
            return (Plato::MechanicsFactory::elastostatics_residual<EvaluationType>(aSpatialDomain, aDataMap, aProblemParams, aPDE));
        }
        else
        {
            THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList")
        }
    }


    /******************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<EvaluationType>>
    createScalarFunctionParabolic(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aParamList,
              std::string              strScalarFunctionType,
              std::string              aStrScalarFunctionName )
    /******************************************************************************/
    {
        THROWERR("Not yet implemented")
    }

    /******************************************************************************//**
     * \brief Create a PLATO scalar function (i.e. optimization criterion)
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap PLATO Analyze physics-based database
     * \param [in] aProblemParams input parameters
     * \param [in] aFuncType scalar function type
     * \param [in] aFuncName scalar function name
    **********************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
    createScalarFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams,
              std::string              aFuncType,
              std::string              aFuncName
    )
    {
        auto tLowerFuncType = Plato::tolower(aFuncType);
        if(tLowerFuncType == "internal elastic energy")
        {
            return Plato::MechanicsFactory::internal_elastic_energy<EvaluationType>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if(tLowerFuncType == "stress p-norm")
        {
            return Plato::MechanicsFactory::stress_p_norm<EvaluationType>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if(tLowerFuncType == "vol avg stress p-norm denominator")
        {
            return Plato::MechanicsFactory::vol_avg_stress_p_norm_denominator<EvaluationType>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if(tLowerFuncType == "effective energy")
        {
            return Plato::MechanicsFactory::effective_energy<EvaluationType>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if(tLowerFuncType == "stress constraint")
        {
            return Plato::MechanicsFactory::stress_constraint_linear<EvaluationType>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if(tLowerFuncType == "stress constraint general")
        {
            return Plato::MechanicsFactory::stress_constraint_general<EvaluationType>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if(tLowerFuncType == "stress constraint quadratic")
        {
            return Plato::MechanicsFactory::stress_constraint_quadratic<EvaluationType>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if(tLowerFuncType == "density penalty")
        {
            return std::make_shared<Plato::IntermediateDensityPenalty<EvaluationType>>
                       (aSpatialDomain, aDataMap, aProblemParams.sublist("Criteria"), aFuncName);
        }
        else if(tLowerFuncType == "volume")
        {
            return Plato::MechanicsFactory::volume_criterion<EvaluationType>
                       (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else
        {
            const std::string tErrorString = std::string("Function '") + tLowerFuncType + "' not implemented yet in steady state mechanics.";
            THROWERR(tErrorString)
        }
    }

#ifdef COMPILE_DEAD_CODE
    /******************************************************************************//**
     * \brief Create a local measure for use in augmented lagrangian quadratic
     * \param [in] aProblemParams input parameters
     * \param [in] aFuncName scalar function name
    **********************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<Plato::AbstractLocalMeasure<EvaluationType,Plato::SimplexMechanics<EvaluationType::SpatialDim>>> 
    createLocalMeasure(Teuchos::ParameterList& aProblemParams, const std::string & aFuncName)
    {
        auto tFunctionSpecs = aProblemParams.sublist("Criteria").sublist(aFuncName);
        auto tLocalMeasure = tFunctionSpecs.get<std::string>("Local Measure", "VonMises");

        if(tLocalMeasure == "VonMises")
        {
            return std::make_shared<VonMisesLocalMeasure<EvaluationType, Plato::SimplexMechanics<EvaluationType::SpatialDim>>>(aProblemParams, "VonMises");
        }
        else if(tLocalMeasure == "TensileEnergyDensity")
        {
            return std::make_shared<TensileEnergyDensityLocalMeasure<EvaluationType, Plato::SimplexMechanics<EvaluationType::SpatialDim>>>
                                                             (aProblemParams, "TensileEnergyDensity");
        }
        else
        {
            THROWERR("Unknown 'Local Measure' specified in 'Plato Problem' ParameterList")
        }
    }
#endif
};
// struct FunctionFactory

} // namespace MechanicsFactory

/******************************************************************************//**
 * \brief Concrete class for use as the SimplexPhysics template argument in
 *        Plato::Elliptic::Problem
**********************************************************************************/
template<Plato::OrdinalType SpaceDimParam>
class Mechanics: public Plato::SimplexMechanics<SpaceDimParam>
{
public:
    typedef Plato::MechanicsFactory::FunctionFactory FunctionFactory;
    using SimplexT = SimplexMechanics<SpaceDimParam>;
    static constexpr Plato::OrdinalType SpaceDim = SpaceDimParam;
};
} // namespace Plato

#endif
