#pragma once

#include <Teuchos_ParameterList.hpp>

#include "PlatoMask.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato {

    using OrdinalList = Plato::ScalarVectorT<Plato::OrdinalType>;

    /******************************************************************************/
    /*!
      \brief class for sequence entries
    */
    template <int mSpatialDim>
    class SequenceStep
    /******************************************************************************/
    {
        std::string mName;
        std::shared_ptr<Plato::Mask<mSpatialDim>> mMask;

      public:
        decltype(mMask) getMask() const {return mMask;}

        template<int mNumDofsPerNode>
        void
        constrainInactiveNodes(
            const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
                  Plato::ScalarVector                  aVector
        ) const
        {
            auto tNodes = mMask->getInactiveNodes();

            if(aMatrix->isBlockMatrix())
            {
                Plato::applyBlockConstraints<mNumDofsPerNode>
                    (aMatrix, aVector, tNodes);
            }
            else
            {
                Plato::applyConstraints<mNumDofsPerNode>
                    (aMatrix, aVector, tNodes);
            }
        }

        /******************************************************************************//**
         * \brief Constructor
         * \param [in] aInputParams SequenceStep definition
        **********************************************************************************/
        SequenceStep(
                  Plato::SpatialModel    & aSpatialModel,
            const Teuchos::ParameterList & aInputParams,
                  std::string              aName
        ) :
            mName(aName)
        {
            Plato::MaskFactory<mSpatialDim> tMaskFactory;
            mMask = tMaskFactory.create(aSpatialModel.Mesh, aInputParams);
        }
    };


    /******************************************************************************/
    /*!
      \brief class for sequence entries
    */
    template <int mSpatialDim>
    class Sequence
    /******************************************************************************/
    {
        std::vector<Plato::SequenceStep<mSpatialDim>> mSteps;

      public:
        const decltype(mSteps) & getSteps() { return mSteps; }

        Sequence(
                  Plato::SpatialModel    & aSpatialModel,
            const Teuchos::ParameterList & aInputParams
        )
        {
            if (aInputParams.isSublist("Sequence"))
            {
                auto tSequenceParams = aInputParams.sublist("Sequence");
                if (!tSequenceParams.isSublist("Steps"))
                {
                    THROWERR("Parsing 'Sequence'. Required 'Steps' list not found");
                }

                auto tStepsParams = tSequenceParams.sublist("Steps");
                for(auto tIndex = tStepsParams.begin(); tIndex != tStepsParams.end(); ++tIndex)
                {
                    const auto & tEntry  = tStepsParams.entry(tIndex);
                    const auto & tMyName = tStepsParams.name(tIndex);

                    if (!tEntry.isList())
                    {
                        THROWERR("Parameter in 'Steps' list not valid.  Expect lists only.");
                    }

                    Teuchos::ParameterList& tStepParams = tStepsParams.sublist(tMyName);
                    mSteps.push_back({aSpatialModel, tStepParams, tMyName});
                }
            }
        }
    };
} // namespace Plato
