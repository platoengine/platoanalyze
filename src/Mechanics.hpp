#ifndef PLATO_MECHANICS_HPP
#define PLATO_MECHANICS_HPP

#include <memory>

#include "parabolic/AbstractScalarFunction.hpp"

#include "elliptic/AbstractScalarFunction.hpp"
#include "elliptic/InternalElasticEnergy.hpp"
#include "elliptic/ElastostaticResidual.hpp"
#include "elliptic/EffectiveEnergy.hpp"
#include "elliptic/StressPNorm.hpp"
#include "elliptic/SurfaceArea.hpp"

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

      if(tLocalMeasure == "VonMises")
      {
          return std::make_shared<VonMisesLocalMeasure<EvaluationType, Plato::SimplexMechanics<EvaluationType::SpatialDim>>>
              (aSpatialDomain, aProblemParams, "VonMises");
      }
      else if(tLocalMeasure == "TensileEnergyDensity")
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
    if(tPenaltyType == "SIMP")
    {
        tOutput = std::make_shared<Plato::Elliptic::ElastostaticResidual<EvaluationType, Plato::MSIMP>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams);
    }
    else
    if(tPenaltyType == "RAMP")
    {
        tOutput = std::make_shared<Plato::Elliptic::ElastostaticResidual<EvaluationType, Plato::RAMP>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams);
    }
    else
    if(tPenaltyType == "Heaviside")
    {
        tOutput = std::make_shared<Plato::Elliptic::ElastostaticResidual<EvaluationType, Plato::Heaviside>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams);
    }
    else
    if(tPenaltyType == "NoPenalty")
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
 * \param [in] aFuncType vector function name
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
    if(tPenaltyType == "SIMP")
    {
        tOutput = std::make_shared<Plato::Elliptic::InternalElasticEnergy<EvaluationType, Plato::MSIMP>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "RAMP")
    {
        tOutput = std::make_shared<Plato::Elliptic::InternalElasticEnergy<EvaluationType, Plato::RAMP>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "Heaviside")
    {
        tOutput = std::make_shared<Plato::Elliptic::InternalElasticEnergy<EvaluationType, Plato::Heaviside>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "NoPenalty")
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
 * \param [in] aFuncType vector function name
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
    if(tPenaltyType == "SIMP")
    {
        tOutput = std::make_shared<Plato::Elliptic::StressPNorm<EvaluationType, Plato::MSIMP>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "RAMP")
    {
        tOutput = std::make_shared<Plato::Elliptic::StressPNorm<EvaluationType, Plato::RAMP>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "Heaviside")
    {
        tOutput = std::make_shared<Plato::Elliptic::StressPNorm<EvaluationType, Plato::Heaviside>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "NoPenalty")
    {
        tOutput = std::make_shared<Plato::Elliptic::StressPNorm<EvaluationType, Plato::NoPenalty>>
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
 * \param [in] aFuncType vector function name
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
    if(tPenaltyType == "SIMP")
    {
        tOutput = std::make_shared<Plato::Elliptic::EffectiveEnergy<EvaluationType, Plato::MSIMP>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "RAMP")
    {
        tOutput = std::make_shared<Plato::Elliptic::EffectiveEnergy<EvaluationType, Plato::RAMP>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "Heaviside")
    {
        tOutput = std::make_shared<Plato::Elliptic::EffectiveEnergy<EvaluationType, Plato::Heaviside>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "NoPenalty")
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
 * \param [in] aFuncType vector function name
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
     * \param [in] aFuncName vector function name
    **********************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<EvaluationType>>
    createVectorFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams,
              std::string              aFuncName)
    {

        if(aFuncName == "Elliptic")
        {
            return (Plato::MechanicsFactory::elastostatics_residual<EvaluationType>(aSpatialDomain, aDataMap, aProblemParams, aFuncName));
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
        if(aFuncType == "Internal Elastic Energy")
        {
            return Plato::MechanicsFactory::internal_elastic_energy<EvaluationType>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if(aFuncType == "Stress P-Norm")
        {
            return Plato::MechanicsFactory::stress_p_norm<EvaluationType>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if(aFuncType == "Effective Energy")
        {
            return Plato::MechanicsFactory::effective_energy<EvaluationType>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if(aFuncType == "Stress Constraint")
        {
            return Plato::MechanicsFactory::stress_constraint_linear<EvaluationType>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if(aFuncType == "Stress Constraint General")
        {
            return Plato::MechanicsFactory::stress_constraint_general<EvaluationType>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if(aFuncType == "Stress Constraint Quadratic")
        {
            return Plato::MechanicsFactory::stress_constraint_quadratic<EvaluationType>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if(aFuncType == "Density Penalty")
        {
            return std::make_shared<Plato::IntermediateDensityPenalty<EvaluationType>>
                       (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else
        {
            return nullptr;
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
