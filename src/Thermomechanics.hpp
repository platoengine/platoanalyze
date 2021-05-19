#ifndef PLATO_THERMOMECHANICS_HPP
#define PLATO_THERMOMECHANICS_HPP

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "Simplex.hpp"
#include "SimplexThermomechanics.hpp"

#include "parabolic/AbstractScalarFunction.hpp"
#include "parabolic/TransientThermomechResidual.hpp"
#include "parabolic/InternalThermoelasticEnergy.hpp"

#include "elliptic/Volume.hpp"
#include "elliptic/AbstractVectorFunction.hpp"
#include "elliptic/ThermoelastostaticResidual.hpp"
#include "elliptic/InternalThermoelasticEnergy.hpp"

#include "AbstractLocalMeasure.hpp"
#include "AnalyzeMacros.hpp"
//#include "TMStressPNorm.hpp"
#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"
#include "AnalyzeMacros.hpp"
#include "Plato_AugLagStressCriterionQuadratic.hpp"
#include "ThermalVonMisesLocalMeasure.hpp"

namespace Plato
{

namespace ThermomechanicsFactory
{
    /******************************************************************************//**
    * \brief Create a local measure for use in augmented lagrangian quadratic
    * \param [in] aProblemParams input parameters
    * \param [in] aFuncName scalar function name
    **********************************************************************************/
    template <typename EvaluationType>
    inline std::shared_ptr<Plato::AbstractLocalMeasure<EvaluationType,Plato::SimplexThermomechanics<EvaluationType::SpatialDim>>> 
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
          return std::make_shared<ThermalVonMisesLocalMeasure<EvaluationType, Plato::SimplexThermomechanics<EvaluationType::SpatialDim>>>
              (aSpatialDomain, aProblemParams, "VonMises");
        }
        else
        {
          THROWERR("Unknown 'Local Measure' specified in 'Plato Problem' ParameterList")
        }
    }

    /******************************************************************************//**
     * \brief Create augmented Lagrangian local constraint criterion with quadratic constraint formulation
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aDataMap PLATO Analyze physics-based database
     * \param [in] aInputParams input parameters
    **********************************************************************************/
    template<typename EvaluationType>
    inline std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
    stress_constraint_quadratic(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aFuncName)
    {
        auto EvalMeasure = Plato::ThermomechanicsFactory::create_local_measure<EvaluationType>(aSpatialDomain, aInputParams, aFuncName);
        using Residual = typename Plato::ResidualTypes<Plato::SimplexThermomechanics<EvaluationType::SpatialDim>>;
        auto PODMeasure = Plato::ThermomechanicsFactory::create_local_measure<Residual>(aSpatialDomain, aInputParams, aFuncName);

        using SimplexT = Plato::SimplexThermomechanics<EvaluationType::SpatialDim>;
        std::shared_ptr<Plato::AugLagStressCriterionQuadratic<EvaluationType,SimplexT>> tOutput;
        tOutput = std::make_shared< Plato::AugLagStressCriterionQuadratic<EvaluationType,SimplexT> >
                    (aSpatialDomain, aDataMap, aInputParams, aFuncName);
        //THROWERR("Not finished implementing this for thermomechanics... need local measure that is compatible.")
        tOutput->setLocalMeasure(EvalMeasure, PODMeasure);
        return (tOutput);
    }

/******************************************************************************/
struct FunctionFactory
{
    /******************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<EvaluationType>>
    createVectorFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aParamList,
              std::string              aStrVectorFunctionType
    )
    /******************************************************************************/
    {

        if(aStrVectorFunctionType == "Elliptic")
        {
            auto tPenaltyParams = aParamList.sublist(aStrVectorFunctionType).sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::Elliptic::ThermoelastostaticResidual<EvaluationType, Plato::MSIMP>>
                         (aSpatialDomain, aDataMap, aParamList, tPenaltyParams);
            }
            else
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::Elliptic::ThermoelastostaticResidual<EvaluationType, Plato::RAMP>>
                         (aSpatialDomain, aDataMap, aParamList, tPenaltyParams);
            }
            else
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::Elliptic::ThermoelastostaticResidual<EvaluationType, Plato::Heaviside>>
                         (aSpatialDomain, aDataMap, aParamList, tPenaltyParams);
            }
            else
            if(tPenaltyType == "NoPenalty")
            {
                return std::make_shared<Plato::Elliptic::ThermoelastostaticResidual<EvaluationType, Plato::NoPenalty>>
                         (aSpatialDomain, aDataMap, aParamList, tPenaltyParams);
            }
            else
            {
                THROWERR("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else
        {
            THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
    }

    /******************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<Plato::Parabolic::AbstractVectorFunction<EvaluationType>>
    createVectorFunctionParabolic(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aParamList,
              std::string              strVectorFunctionType
    )
    /******************************************************************************/
    {
        if( strVectorFunctionType == "Parabolic" )
        {
            auto tPenaltyParams = aParamList.sublist(strVectorFunctionType).sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type");
            if( tPenaltyType == "SIMP" )
            {
                return std::make_shared<Plato::Parabolic::TransientThermomechResidual<EvaluationType, Plato::MSIMP>>
                         (aSpatialDomain, aDataMap, aParamList, tPenaltyParams);
            } else
            if( tPenaltyType == "RAMP" )
            {
                return std::make_shared<Plato::Parabolic::TransientThermomechResidual<EvaluationType, Plato::RAMP>>
                         (aSpatialDomain, aDataMap, aParamList, tPenaltyParams);
            } else
            if( tPenaltyType == "Heaviside" )
            {
                return std::make_shared<Plato::Parabolic::TransientThermomechResidual<EvaluationType, Plato::Heaviside>>
                         (aSpatialDomain, aDataMap, aParamList, tPenaltyParams);
            } else {
                THROWERR("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else
        {
            THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
    }
    /******************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
    createScalarFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aParamList, 
              std::string              aStrScalarFunctionType,
              std::string              aStrScalarFunctionName
    )
    /******************************************************************************/
    {

        auto tPenaltyParams = aParamList.sublist("Criteria").sublist(aStrScalarFunctionName).sublist("Penalty Function");
        std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");

        if(aStrScalarFunctionType == "Internal Thermoelastic Energy")
        {
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::Elliptic::InternalThermoelasticEnergy<EvaluationType, Plato::MSIMP>>
                         (aSpatialDomain, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else 
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::Elliptic::InternalThermoelasticEnergy<EvaluationType, Plato::RAMP>>
                         (aSpatialDomain, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else 
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::Elliptic::InternalThermoelasticEnergy<EvaluationType, Plato::Heaviside>>
                         (aSpatialDomain, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else
            {
                THROWERR("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else
        if(aStrScalarFunctionType == "Stress Constraint Quadratic")
        {
            return (Plato::ThermomechanicsFactory::stress_constraint_quadratic<EvaluationType>
                   (aSpatialDomain, aDataMap, aParamList, aStrScalarFunctionName));
        }
        else
        if(aStrScalarFunctionType == "Volume" )
        {
            if(tPenaltyType == "SIMP" )
            {
                return std::make_shared<Plato::Elliptic::Volume<EvaluationType, Plato::MSIMP>>
                         (aSpatialDomain, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else
            if(tPenaltyType == "RAMP" )
            {
                return std::make_shared<Plato::Elliptic::Volume<EvaluationType, Plato::RAMP>>
                         (aSpatialDomain, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else
            if(tPenaltyType == "Heaviside" )
            {
                return std::make_shared<Plato::Elliptic::Volume<EvaluationType, Plato::Heaviside>>
                         (aSpatialDomain, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else
            {
                throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else 
#ifdef NOPE
        if(aStrScalarFunctionType == "Stress P-Norm")
        {
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::TMStressPNorm<EvaluationType, Plato::MSIMP>>
                         (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else 
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::TMStressPNorm<EvaluationType, Plato::RAMP>>
                         (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else 
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::TMStressPNorm<EvaluationType, Plato::Heaviside>>
                         (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else
            {
                THROWERR("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else
#endif
        {
            THROWERR("Unknown 'Objective' specified in 'Plato Problem' ParameterList");
        }
    }
    /******************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<EvaluationType>>
    createScalarFunctionParabolic(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aParamList,
              std::string              aStrScalarFunctionType,
              std::string              aStrScalarFunctionName
    )
    /******************************************************************************/
    {
        auto tPenaltyParams = aParamList.sublist("Criteria").sublist(aStrScalarFunctionName).sublist("Penalty Function");
        std::string tPenaltyType = tPenaltyParams.get<std::string>("Type");
        if( aStrScalarFunctionType == "Internal Thermoelastic Energy" )
        {
            if( tPenaltyType == "SIMP" )
            {
                return std::make_shared<Plato::Parabolic::InternalThermoelasticEnergy<EvaluationType, Plato::MSIMP>>
                     (aSpatialDomain, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else
            if( tPenaltyType == "RAMP" )
            {
                return std::make_shared<Plato::Parabolic::InternalThermoelasticEnergy<EvaluationType, Plato::RAMP>>
                     (aSpatialDomain, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else
            if( tPenaltyType == "Heaviside" )
            {
                return std::make_shared<Plato::Parabolic::InternalThermoelasticEnergy<EvaluationType, Plato::Heaviside>>
                     (aSpatialDomain, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else
            {
                THROWERR("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else
        {
            THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
    }

}; // struct FunctionFactory

} // namespace ThermomechanicsFactory


/****************************************************************************//**
 * \brief Concrete class for use as the SimplexPhysics template argument in
 *        Plato::Elliptic::Problem and Plato::Parabolic::Problem
 *******************************************************************************/
template<Plato::OrdinalType SpaceDimParam>
class Thermomechanics: public Plato::SimplexThermomechanics<SpaceDimParam>
{
public:
    typedef Plato::ThermomechanicsFactory::FunctionFactory FunctionFactory;
    using SimplexT = SimplexThermomechanics<SpaceDimParam>;
    static constexpr Plato::OrdinalType SpaceDim = SpaceDimParam;
};
} // namespace Plato

#endif
