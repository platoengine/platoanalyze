#ifndef PLATO_MECHANICS_HPP
#define PLATO_MECHANICS_HPP

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "parabolic/AbstractScalarFunction.hpp"

#include "elliptic/AbstractScalarFunction.hpp"
#include "elliptic/InternalElasticEnergy.hpp"
#include "elliptic/ElastostaticResidual.hpp"
#include "elliptic/EffectiveEnergy.hpp"
#include "elliptic/StressPNorm.hpp"
#include "elliptic/SurfaceArea.hpp"

//TODO #include "Plato_AugLagStressCriterionQuadratic.hpp"
//TODO #include "Plato_AugLagStressCriterionGeneral.hpp"
//TODO #include "Plato_AugLagStressCriterion.hpp"
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
   * \param [in] aInputParams input parameters
   * \param [in] aFuncName scalar function name
  **********************************************************************************/
  template <typename EvaluationType>
  inline std::shared_ptr<Plato::AbstractLocalMeasure<EvaluationType,Plato::SimplexMechanics<EvaluationType::SpatialDim>>> 
  create_local_measure(Teuchos::ParameterList& aInputParams, const std::string & aFuncName)
  {
      auto tFunctionSpecs = aInputParams.sublist(aFuncName);
      auto tLocalMeasure = tFunctionSpecs.get<std::string>("Local Measure", "VonMises");

      if(tLocalMeasure == "VonMises")
      {
          return std::make_shared<VonMisesLocalMeasure<EvaluationType, Plato::SimplexMechanics<EvaluationType::SpatialDim>>>(aInputParams, "VonMises");
      }
      else if(tLocalMeasure == "TensileEnergyDensity")
      {
          return std::make_shared<TensileEnergyDensityLocalMeasure<EvaluationType, Plato::SimplexMechanics<EvaluationType::SpatialDim>>>
                                                           (aInputParams, "TensileEnergyDensity");
      }
      else
      {
          THROWERR("Unknown 'Local Measure' specified in 'Plato Problem' ParameterList")
      }
  }

/******************************************************************************//**
 * \brief Create elastostatics residual equation
 * \param [in] aMesh mesh database
 * \param [in] aMeshSets side sets database
 * \param [in] aDataMap PLATO Analyze physics-based database
 * \param [in] aInputParams input parameters
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<EvaluationType>>
elastostatics_residual(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aInputParams,
          std::string              aFuncName
)
{
    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<EvaluationType>> tOutput;
    auto tPenaltyParams = aInputParams.sublist(aFuncName).sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    if(tPenaltyType == "SIMP")
    {
        tOutput = std::make_shared<Plato::Elliptic::ElastostaticResidual<EvaluationType, Plato::MSIMP>>
                    (aSpatialDomain, aDataMap, aInputParams, tPenaltyParams);
    }
    else
    if(tPenaltyType == "RAMP")
    {
        tOutput = std::make_shared<Plato::Elliptic::ElastostaticResidual<EvaluationType, Plato::RAMP>>
                    (aSpatialDomain, aDataMap, aInputParams, tPenaltyParams);
    }
    else
    if(tPenaltyType == "Heaviside")
    {
        tOutput = std::make_shared<Plato::Elliptic::ElastostaticResidual<EvaluationType, Plato::Heaviside>>
                    (aSpatialDomain, aDataMap, aInputParams, tPenaltyParams);
    }
    else
    if(tPenaltyType == "NoPenalty")
    {
        tOutput = std::make_shared<Plato::Elliptic::ElastostaticResidual<EvaluationType, Plato::NoPenalty>>
                    (aSpatialDomain, aDataMap, aInputParams, tPenaltyParams);
    }
    return (tOutput);
}
// function elastostatics_residual

/******************************************************************************//**
 * \brief Create augmented Lagrangian stress constraint criterion tailored for linear problems
 * \param [in] aMesh mesh database
 * \param [in] aMeshSets side sets database
 * \param [in] aDataMap PLATO Analyze physics-based database
 * \param [in] aInputParams input parameters
**********************************************************************************/
/* TODO
template<typename EvaluationType>
inline std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
stress_constraint_linear(Omega_h::Mesh& aMesh,
                         Omega_h::MeshSets& aMeshSets,
                         Plato::DataMap& aDataMap,
                         Teuchos::ParameterList & aInputParams,
                         std::string & aFuncName)
{
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>> tOutput;
    tOutput = std::make_shared< Plato::AugLagStressCriterion<EvaluationType> >
                (aMesh, aMeshSets, aDataMap, aInputParams, aFuncName);
    return (tOutput);
}
*/

/******************************************************************************//**
 * \brief Create augmented Lagrangian stress constraint criterion tailored for general problems
 * \param [in] aMesh mesh database
 * \param [in] aMeshSets side sets database
 * \param [in] aDataMap PLATO Analyze physics-based database
 * \param [in] aInputParams input parameters
**********************************************************************************/
/* TODO
template<typename EvaluationType>
inline std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
stress_constraint_general(Omega_h::Mesh& aMesh,
                          Omega_h::MeshSets& aMeshSets,
                          Plato::DataMap& aDataMap,
                          Teuchos::ParameterList & aInputParams,
                          std::string & aFuncName)
{
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>> tOutput;
    tOutput = std::make_shared <Plato::AugLagStressCriterionGeneral<EvaluationType> >
                (aMesh, aMeshSets, aDataMap, aInputParams, aFuncName);
    return (tOutput);
}
*/


/******************************************************************************//**
 * \brief Create augmented Lagrangian local constraint criterion with quadratic constraint formulation
 * \param [in] aMesh mesh database
 * \param [in] aMeshSets side sets database
 * \param [in] aDataMap PLATO Analyze physics-based database
 * \param [in] aInputParams input parameters
**********************************************************************************/
/* TODO
template<typename EvaluationType>
inline std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
stress_constraint_quadratic(Omega_h::Mesh& aMesh,
                            Omega_h::MeshSets& aMeshSets,
                            Plato::DataMap& aDataMap,
                            Teuchos::ParameterList & aInputParams,
                            std::string & aFuncName)
{
    auto EvalMeasure = Plato::MechanicsFactory::create_local_measure<EvaluationType>(aInputParams, aFuncName);
    using Residual = typename Plato::ResidualTypes<Plato::SimplexMechanics<EvaluationType::SpatialDim>>;
    auto PODMeasure = Plato::MechanicsFactory::create_local_measure<Residual>(aInputParams, aFuncName);

    using SimplexT = Plato::SimplexMechanics<EvaluationType::SpatialDim>;
    std::shared_ptr<Plato::AugLagStressCriterionQuadratic<EvaluationType,SimplexT>> tOutput;
    tOutput = std::make_shared< Plato::AugLagStressCriterionQuadratic<EvaluationType,SimplexT> >
                (aMesh, aMeshSets, aDataMap, aInputParams, aFuncName);

    tOutput->setLocalMeasure(EvalMeasure, PODMeasure);
    return (tOutput);
}
*/


/******************************************************************************//**
 * \brief Create internal elastic energy criterion
 * \param [in] aMesh mesh database
 * \param [in] aMeshSets side sets database
 * \param [in] aDataMap PLATO Analyze physics-based database
 * \param [in] aInputParams input parameters
 * \param [in] aFuncType vector function name
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
internal_elastic_energy(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aInputParams,
          std::string            & aFuncName
)
{
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>> tOutput;
    auto tPenaltyParams = aInputParams.sublist(aFuncName).sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    if(tPenaltyType == "SIMP")
    {
        tOutput = std::make_shared<Plato::Elliptic::InternalElasticEnergy<EvaluationType, Plato::MSIMP>>
                    (aSpatialDomain, aDataMap, aInputParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "RAMP")
    {
        tOutput = std::make_shared<Plato::Elliptic::InternalElasticEnergy<EvaluationType, Plato::RAMP>>
                    (aSpatialDomain, aDataMap, aInputParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "Heaviside")
    {
        tOutput = std::make_shared<Plato::Elliptic::InternalElasticEnergy<EvaluationType, Plato::Heaviside>>
                    (aSpatialDomain, aDataMap, aInputParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "NoPenalty")
    {
        tOutput = std::make_shared<Plato::Elliptic::InternalElasticEnergy<EvaluationType, Plato::NoPenalty>>
                    (aSpatialDomain, aDataMap, aInputParams, tPenaltyParams, aFuncName);
    }
    return (tOutput);
}
// function internal_elastic_energy

/******************************************************************************//**
 * \brief Create stress p-norm criterion
 * \param [in] aMesh mesh database
 * \param [in] aMeshSets side sets database
 * \param [in] aDataMap PLATO Analyze physics-based database
 * \param [in] aInputParams input parameters
 * \param [in] aFuncType vector function name
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
stress_p_norm(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aInputParams,
          std::string            & aFuncName
)
{
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>> tOutput;
    auto tPenaltyParams = aInputParams.sublist(aFuncName).sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    if(tPenaltyType == "SIMP")
    {
        tOutput = std::make_shared<Plato::Elliptic::StressPNorm<EvaluationType, Plato::MSIMP>>
                    (aSpatialDomain, aDataMap, aInputParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "RAMP")
    {
        tOutput = std::make_shared<Plato::Elliptic::StressPNorm<EvaluationType, Plato::RAMP>>
                    (aSpatialDomain, aDataMap, aInputParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "Heaviside")
    {
        tOutput = std::make_shared<Plato::Elliptic::StressPNorm<EvaluationType, Plato::Heaviside>>
                    (aSpatialDomain, aDataMap, aInputParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "NoPenalty")
    {
        tOutput = std::make_shared<Plato::Elliptic::StressPNorm<EvaluationType, Plato::NoPenalty>>
                    (aSpatialDomain, aDataMap, aInputParams, tPenaltyParams, aFuncName);
    }
    return (tOutput);
}
// function stress_p_norm

/******************************************************************************//**
 * \brief Create effective energy criterion
 * \param [in] aMesh mesh database
 * \param [in] aMeshSets side sets database
 * \param [in] aDataMap PLATO Analyze physics-based database
 * \param [in] aInputParams input parameters
 * \param [in] aFuncType vector function name
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
effective_energy(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aInputParams,
          std::string            & aFuncName
)
{
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>> tOutput;
    auto tPenaltyParams = aInputParams.sublist(aFuncName).sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    if(tPenaltyType == "SIMP")
    {
        tOutput = std::make_shared<Plato::Elliptic::EffectiveEnergy<EvaluationType, Plato::MSIMP>>
                    (aSpatialDomain, aDataMap, aInputParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "RAMP")
    {
        tOutput = std::make_shared<Plato::Elliptic::EffectiveEnergy<EvaluationType, Plato::RAMP>>
                    (aSpatialDomain, aDataMap, aInputParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "Heaviside")
    {
        tOutput = std::make_shared<Plato::Elliptic::EffectiveEnergy<EvaluationType, Plato::Heaviside>>
                    (aSpatialDomain, aDataMap, aInputParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "NoPenalty")
    {
        tOutput = std::make_shared<Plato::Elliptic::EffectiveEnergy<EvaluationType, Plato::NoPenalty>>
                    (aSpatialDomain, aDataMap, aInputParams, tPenaltyParams, aFuncName);
    }
    return (tOutput);
}
// function effective_energy

/******************************************************************************//**
 * \brief Factory for linear mechanics problem
 * @brief Create surface area scalar function
 * @param [in] aMesh mesh database
 * @param [in] aMeshSets side sets database
 * @param [in] aDataMap PLATO Analyze physics-based database
 * @param [in] aInputParams input parameters
 * @param [in] aFuncType vector function name
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
surface_area(Omega_h::Mesh& aMesh,
       Omega_h::MeshSets& aMeshSets,
       Plato::DataMap& aDataMap,
       Teuchos::ParameterList & aInputParams,
       std::string & aFuncName)
{
    return std::make_shared<Plato::Elliptic::SurfaceArea<EvaluationType>>(aMesh, aMeshSets, aDataMap, aInputParams, aFuncName);
}
// function surface area

/******************************************************************************//**
 * @brief Factory for linear mechanics problem
**********************************************************************************/
struct FunctionFactory
{
    /******************************************************************************//**
     * \brief Create a PLATO vector function (i.e. residual equation)
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aDataMap PLATO Analyze physics-based database
     * \param [in] aInputParams input parameters
     * \param [in] aFuncName vector function name
    **********************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<EvaluationType>>
    createVectorFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aInputParams,
              std::string              aFuncName)
    {

        if(aFuncName == "Elliptic")
        {
            return (Plato::MechanicsFactory::elastostatics_residual<EvaluationType>(aSpatialDomain, aDataMap, aInputParams, aFuncName));
        }
        else
        {
            THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList")
        }
    }


    /******************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<EvaluationType>>
    createScalarFunctionParabolic(Omega_h::Mesh& aMesh,
                            Omega_h::MeshSets& aMeshSets,
                            Plato::DataMap& aDataMap,
                            Teuchos::ParameterList& aParamList,
                            std::string strScalarFunctionType,
                            std::string aStrScalarFunctionName )
    /******************************************************************************/
    {
        THROWERR("Not yet implemented")
    }

    /******************************************************************************//**
     * \brief Create a PLATO scalar function (i.e. optimization criterion)
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aDataMap PLATO Analyze physics-based database
     * \param [in] aInputParams input parameters
     * \param [in] aFuncName scalar function name
    **********************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
    createScalarFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aInputParams,
              std::string              aFuncType,
              std::string              aFuncName
    )
    {
        if(aFuncType == "Internal Elastic Energy")
        {
            return Plato::MechanicsFactory::internal_elastic_energy<EvaluationType>
                       (aSpatialDomain, aDataMap, aInputParams, aFuncName);
        }
        else if(aFuncType == "Stress P-Norm")
        {
            return Plato::MechanicsFactory::stress_p_norm<EvaluationType>
                       (aSpatialDomain, aDataMap, aInputParams, aFuncName);
        }
        else if(aFuncType == "Effective Energy")
        {
            return Plato::MechanicsFactory::effective_energy<EvaluationType>
                       (aSpatialDomain, aDataMap, aInputParams, aFuncName);
        }
/* TODO
        else if(aFuncType == "Stress Constraint")
        {
            return (Plato::MechanicsFactory::stress_constraint_linear<EvaluationType>(aMesh, aMeshSets, aDataMap, aInputParams, aFuncName));
        }
        else if(aFuncType == "Stress Constraint General")
        {
            return (Plato::MechanicsFactory::stress_constraint_general<EvaluationType>(aMesh, aMeshSets, aDataMap, aInputParams, aFuncName));
        }
        else if(aFuncType == "Stress Constraint Quadratic")
        {
            return (Plato::MechanicsFactory::stress_constraint_quadratic<EvaluationType>(aMesh, aMeshSets, aDataMap, aInputParams, aFuncName));
        }
*/
        else if(aFuncType == "Density Penalty")
        {
            return std::make_shared<Plato::IntermediateDensityPenalty<EvaluationType>>
                       (aSpatialDomain, aDataMap, aInputParams, aFuncName);
        }
        else
        {
            THROWERR("Unknown 'Objective' specified in 'Plato Problem' ParameterList")
        }
    }

    /******************************************************************************//**
     * \brief Create a local measure for use in augmented lagrangian quadratic
     * \param [in] aInputParams input parameters
     * \param [in] aFuncName scalar function name
    **********************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<Plato::AbstractLocalMeasure<EvaluationType,Plato::SimplexMechanics<EvaluationType::SpatialDim>>> 
    createLocalMeasure(Teuchos::ParameterList& aInputParams, const std::string & aFuncName)
    {
        auto tFunctionSpecs = aInputParams.sublist(aFuncName);
        auto tLocalMeasure = tFunctionSpecs.get<std::string>("Local Measure", "VonMises");

        if(tLocalMeasure == "VonMises")
        {
            return std::make_shared<VonMisesLocalMeasure<EvaluationType, Plato::SimplexMechanics<EvaluationType::SpatialDim>>>(aInputParams, "VonMises");
        }
        else if(tLocalMeasure == "TensileEnergyDensity")
        {
            return std::make_shared<TensileEnergyDensityLocalMeasure<EvaluationType, Plato::SimplexMechanics<EvaluationType::SpatialDim>>>
                                                             (aInputParams, "TensileEnergyDensity");
        }
        else
        {
            THROWERR("Unknown 'Local Measure' specified in 'Plato Problem' ParameterList")
        }
    }
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
