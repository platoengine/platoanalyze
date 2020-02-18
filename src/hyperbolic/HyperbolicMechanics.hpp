#ifndef PLATO_HYPERBOLIC_MECHANICS_HPP
#define PLATO_HYPERBOLIC_MECHANICS_HPP

#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "SimplexMechanics.hpp"
#include "ScalarFunctionBase.hpp"
#include "hyperbolic/ElastomechanicsResidual.hpp"

namespace Plato
{
  namespace Hyperbolic
  {
    struct FunctionFactory
    {
      /******************************************************************************/
      template <typename EvaluationType>
      std::shared_ptr<AbstractVectorFunctionHyperbolic<EvaluationType>>
      createVectorFunctionHyperbolic(Omega_h::Mesh& aMesh,
                              Omega_h::MeshSets& aMeshSets,
                              Plato::DataMap& aDataMap,
                              Teuchos::ParameterList& aParamList,
                              std::string strVectorFunctionType )
      /******************************************************************************/
      {
        if( strVectorFunctionType == "Hyperbolic" )
        {
            auto penaltyParams = aParamList.sublist(strVectorFunctionType).sublist("Penalty Function");
            std::string penaltyType = penaltyParams.get<std::string>("Type");
            if( penaltyType == "SIMP" )
            {
                return std::make_shared<TransientMechanicsResidual<EvaluationType, Plato::MSIMP>>
                         (aMesh,aMeshSets,aDataMap,aParamList,penaltyParams);
            } else
            if( penaltyType == "RAMP" )
            {
                return std::make_shared<TransientMechanicsResidual<EvaluationType, Plato::RAMP>>
                         (aMesh,aMeshSets,aDataMap,aParamList,penaltyParams);
            } else
            if( penaltyType == "Heaviside" )
            {
                return std::make_shared<TransientMechanicsResidual<EvaluationType, Plato::Heaviside>>
                         (aMesh,aMeshSets,aDataMap,aParamList,penaltyParams);
            } else {
                THROWERR("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else
        {
            THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
      }
    };

    /******************************************************************************//**
     * @brief Concrete class for use as the SimplexPhysics template argument in
     *        EllipticProblem
    **********************************************************************************/
    template<Plato::OrdinalType SpaceDimParam>
    class Mechanics: public Plato::SimplexMechanics<SpaceDimParam>
    {
    public:
        typedef Plato::Hyperbolic::FunctionFactory FunctionFactory;
        using SimplexT = SimplexMechanics<SpaceDimParam>;
        static constexpr Plato::OrdinalType SpaceDim = SpaceDimParam;
    };
  } // namespace Hyperbolic

} // namespace Plato

#endif
