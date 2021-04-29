#pragma once

#include "Simplex.hpp"

#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"

#include "geometric/Volume.hpp"
#include "geometric/GeometryMisfit.hpp"
#include "geometric/AbstractScalarFunction.hpp"

namespace Plato {

namespace GeometryFactory {
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
struct FunctionFactory{
/******************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<Plato::Geometric::AbstractScalarFunction<EvaluationType>>
    createScalarFunction( 
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aParamList,
              std::string              strScalarFunctionType,
              std::string              strScalarFunctionName
    )
    {
        if( strScalarFunctionType == "Volume" )
        {
            auto penaltyParams = aParamList.sublist("Criteria").sublist(strScalarFunctionName).sublist("Penalty Function");
            std::string penaltyType = penaltyParams.get<std::string>("Type");
            if( penaltyType == "SIMP" )
            {
                return std::make_shared<Plato::Geometric::Volume<EvaluationType, Plato::MSIMP>>
                   (aSpatialDomain, aDataMap, aParamList, penaltyParams, strScalarFunctionName);
            }
            else
            if( penaltyType == "RAMP" )
            {
                return std::make_shared<Plato::Geometric::Volume<EvaluationType, Plato::RAMP>>
                   (aSpatialDomain, aDataMap, aParamList, penaltyParams, strScalarFunctionName);
            }
            else
            if( penaltyType == "Heaviside" )
            {
                return std::make_shared<Plato::Geometric::Volume<EvaluationType, Plato::Heaviside>>
                   (aSpatialDomain, aDataMap, aParamList, penaltyParams, strScalarFunctionName);
            }
            else
            {
                throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        } else
        if( strScalarFunctionType == "Geometry Misfit" )
        {
            return std::make_shared<Plato::Geometric::GeometryMisfit<EvaluationType>>
               (aSpatialDomain, aDataMap, aParamList, strScalarFunctionName);
        }
        else
        {
            throw std::runtime_error("Unknown 'Objective' specified in 'Plato Problem' ParameterList");
        }
    }
};

} // namespace GeometryFactory

template <Plato::OrdinalType SpaceDimParam>
class Geometrical : public Plato::Simplex<SpaceDimParam> {
  public:
    typedef Plato::GeometryFactory::FunctionFactory<SpaceDimParam> FunctionFactory;
    using SimplexT = Simplex<SpaceDimParam>;
    static constexpr Plato::OrdinalType SpaceDim = SpaceDimParam;
    static constexpr Plato::OrdinalType mNumControl = 1;
};
// class Geometrical

} //namespace Plato
