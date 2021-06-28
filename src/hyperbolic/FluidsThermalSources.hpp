/*
 * FluidsThermalSources.hpp
 *
 *  Created on: June 17, 2021
 */

#pragma once

#include "hyperbolic/FluidsThermalSourceFactory.hpp"

namespace Plato
{

namespace Fluids
{

template<typename PhysicsT, typename EvaluationT>
class ThermalSources
{
private:
    // set local ad type
    using ResultT = typename EvaluationT::ResultScalarType; /*!< result FAD evaluation type */

    // set local member data
    std::vector<std::shared_ptr<Plato::AbstractVolumetricSource<PhysicsT, EvaluationT>>> mSources;  /*!< volumetric source list */

public:
    ThermalSources() : 
        mSources()
    {}

    const decltype(mSources)& 
    sources() const
    {
        return mSources;
    }

    bool empty() const
    {
        return mSources.empty(); 
    }

    void initializeThermalSources
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs)
    {
        this->allocateThermalSources(aDomain, aDataMap, aInputs);
    }

    void initializeStabilizedThermalSources
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs)
    {
        this->allocateStabilizedThermalSources(aDomain, aDataMap, aInputs);
    }

    void evaluate
    (const Plato::WorkSets & aWorkSets, 
     Plato::ScalarMultiVectorT<ResultT> &aResultWS,
     Plato::Scalar aMultiplier = 1.0) const
    {
        for (const auto &tSource : mSources)
        {
            tSource->evaluate(aWorkSets, aResultWS, aMultiplier);
        }
    }

private:
    void allocateThermalSources
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs)
    {
        if( aInputs.isSublist("Thermal Sources") )
        {
            auto tThermalSourcesParamList = aInputs.sublist("Thermal Sources");
            for (Teuchos::ParameterList::ConstIterator tItr = tThermalSourcesParamList.begin(); tItr != tThermalSourcesParamList.end(); ++tItr)
            {
                const Teuchos::ParameterEntry &tEntry = tThermalSourcesParamList.entry(tItr);
                if ( !tEntry.isList() )
                {
                    THROWERR("Parameter in 'Thermal Sources' block not valid. Expects a Parameter List only.")
                }

                std::string tName = tThermalSourcesParamList.name(tItr);
                if(tThermalSourcesParamList.isSublist(tName) == false)
                {
                    THROWERR(std::string("Parameter Sublist '") + tName.c_str() + "' is NOT defined in 'Thermal Sources' block.")
                }
                Teuchos::ParameterList &tSubList = tThermalSourcesParamList.sublist(tName);

                if(tSubList.isParameter("Type") == false)
                {
                    THROWERR(std::string("'Type' Keyword in Parameter Sublist '") + tName.c_str() + "' is NOT defined.")
                }

                auto tFuncType = tSubList.get<std::string>("Type");
                Plato::Fluids::ThermalSourceFactory tFactory;
                auto tSource = tFactory.template createThermalSource<PhysicsT, EvaluationT>(tFuncType, tName, aDomain, aDataMap, aInputs);
                mSources.push_back(tSource);
            }
        }
    }

    void allocateStabilizedThermalSources
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs)
    {
        if( aInputs.isSublist("Thermal Sources") )
        {
            auto tThermalSourcesParamList = aInputs.sublist("Thermal Sources");
            for (Teuchos::ParameterList::ConstIterator tItr = tThermalSourcesParamList.begin(); tItr != tThermalSourcesParamList.end(); ++tItr)
            {
                const Teuchos::ParameterEntry &tEntry = tThermalSourcesParamList.entry(tItr);
                if ( !tEntry.isList() )
                {
                    THROWERR("Parameter in 'Thermal Sources' block not valid. Expects a Parameter List only.")
                }

                std::string tName = tThermalSourcesParamList.name(tItr);
                if(tThermalSourcesParamList.isSublist(tName) == false)
                {
                    THROWERR(std::string("Parameter Sublist '") + tName.c_str() + "' is NOT defined in 'Thermal Sources' block.")
                }
                Teuchos::ParameterList &tSubList = tThermalSourcesParamList.sublist(tName);

                if(tSubList.isParameter("Type") == false)
                {
                    THROWERR(std::string("'Type' Keyword in Parameter Sublist '") + tName.c_str() + "' is NOT defined.")
                }

                auto tFuncType = tSubList.get<std::string>("Type");
                Plato::Fluids::ThermalSourceFactory tFactory;
                auto tSource = tFactory.template createStabilizedThermalSource<PhysicsT, EvaluationT>(tFuncType, tName, aDomain, aDataMap, aInputs);
                mSources.push_back(tSource);
            }
        }
    }
};
// class ThermalSources

}
// namespace Fluids

}
// namespace Plato

#include "hyperbolic/IncompressibleFluids.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::ThermalSources, Plato::EnergyConservation, Plato::SimplexFluids, 1, 1)
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::ThermalSources, Plato::IncompressibleFluids, Plato::SimplexFluids, 1, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::ThermalSources, Plato::EnergyConservation, Plato::SimplexFluids, 2, 1)
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::ThermalSources, Plato::IncompressibleFluids, Plato::SimplexFluids, 2, 1)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::ThermalSources, Plato::EnergyConservation, Plato::SimplexFluids, 3, 1)
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::ThermalSources, Plato::IncompressibleFluids, Plato::SimplexFluids, 3, 1)
#endif
