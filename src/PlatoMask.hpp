#pragma once

#include <Omega_h_mesh.hpp>
#include <Teuchos_ParameterList.hpp>

#include "ImplicitFunctors.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato {

    template <int mSpaceDim>
    struct BoxLimits {
      Plato::Scalar mMaximum[mSpaceDim];
      Plato::Scalar mMinimum[mSpaceDim];
      std::string mMaxKeywords[3];
      std::string mMinKeywords[3];

      BoxLimits(
          const Teuchos::ParameterList& aParams
      ) :
          mMaxKeywords{"Maximum X", "Maximum Y", "Maximum Z"},
          mMinKeywords{"Minimum X", "Minimum Y", "Minimum Z"}
      {
          for (Plato::OrdinalType tDim=0; tDim<mSpaceDim; tDim++)
          {
              auto tMaxKeyword = mMaxKeywords[tDim];
              if (aParams.isType<Plato::Scalar>(tMaxKeyword))
              {
                  mMaximum[tDim] = aParams.get<Plato::Scalar>(tMaxKeyword);
              }
              else
              {
                  mMaximum[tDim] = 1e12;
              }
              auto tMinKeyword = mMinKeywords[tDim];
              if (aParams.isType<Plato::Scalar>(tMinKeyword))
              {
                  mMinimum[tDim] = aParams.get<Plato::Scalar>(tMinKeyword);
              }
              else
              {
                  mMinimum[tDim] = -1e12;
              }
          }
      }
    };

    class ConstructivePrimitive
    {
      public:
        virtual void apply( LocalOrdinalVector aCellMask, Plato::ScalarMultiVector aCellCenters ) const = 0;
    };

    class BrickPrimitive : public ConstructivePrimitive
    {
        BoxLimits<3> mLimits;
        Plato::OrdinalType mOperation;

      public:
        BrickPrimitive( Teuchos::ParameterList& aParams );
        void apply( LocalOrdinalVector aCellMask, Plato::ScalarMultiVector aCellCenters ) const override;
    };

    /******************************************************************************/
    /*!
      \brief class for sequence entries
    */
    template <int mSpaceDim>
    class Mask
    /******************************************************************************/
    {

      protected:
        LocalOrdinalVector mCellMask;
        LocalOrdinalVector mNodeMask;

      public:
        /******************************************************************************//**
         * \brief Compute node mask from element mask
         * \param [in] aMesh Omega_h mesh
        **********************************************************************************/
        void
        computeNodeMask(
            Omega_h::Mesh & aMesh
        )
        {
            Kokkos::deep_copy(mNodeMask, 0.0);

            NodeOrdinal<mSpaceDim> tNodeOrdinalFunctor(&aMesh);

            auto tCellMask = mCellMask;
            auto tNodeMask = mNodeMask;

            auto tNumCells = mCellMask.extent(0);
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
            {
                if (tCellMask(aCellOrdinal) == 1)
                {
                    for (Plato::OrdinalType tNode=0; tNode<mSpaceDim+1; tNode++)
                    {
                        auto tNodeOrdinal = tNodeOrdinalFunctor(aCellOrdinal, tNode);
                        tNodeMask(tNodeOrdinal) = 1;
                    }
                }
            }, "compute node mask");
        }

        /******************************************************************************//**
         * \brief get location of cell centers in physical space
         * \param [in] aMesh Omega_h mesh
        **********************************************************************************/
        Plato::ScalarMultiVector
        getCellCenters(
            Omega_h::Mesh & aMesh
        )
        {
            Plato::NodeCoordinate<mSpaceDim> tNodeCoordsFunctor(&aMesh);

            auto tNumCells = aMesh.nelems();
            Plato::ScalarMultiVector tCellCenters("cell centers", tNumCells, mSpaceDim);

            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells),
            LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
            {
                for (Plato::OrdinalType tNode=0; tNode<mSpaceDim+1; tNode++)
                {
                    for (Plato::OrdinalType tDim=0; tDim<mSpaceDim; tDim++)
                    {
                        tCellCenters(aCellOrdinal, tDim) += tNodeCoordsFunctor(aCellOrdinal, tNode, tDim);
                    }
                }
                for (Plato::OrdinalType tDim=0; tDim<mSpaceDim; tDim++)
                {
                    tCellCenters(aCellOrdinal, tDim) /= (mSpaceDim+1);
                }
            }, "get cell centers");

            return tCellCenters;
        }

        Mask(
            const Omega_h::Mesh & aMesh
        ) :
            mCellMask("cell mask", aMesh.nelems()),
            mNodeMask("node mask", aMesh.nverts()){}

        decltype(mCellMask) cellMask() const {return mCellMask;}
        decltype(mNodeMask) nodeMask() const {return mNodeMask;}

        /******************************************************************************//**
         * \brief Compute node mask from element mask
         * \param [in] aMesh Omega_h mesh
        **********************************************************************************/
        LocalOrdinalVector
        getInactiveNodes(
        ) const
        {
            using OrdinalT = Plato::OrdinalType;

            auto tNumEntries = mNodeMask.extent(0);

            // how many zeros in the mask?
            Plato::OrdinalType tSum(0);
            auto tNodeMask = mNodeMask;
            Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0,tNumEntries),
            LAMBDA_EXPRESSION(const Plato::OrdinalType& aOrdinal, Plato::OrdinalType & aUpdate)
            {
              aUpdate += tNodeMask(aOrdinal);
            }, tSum);

            auto tNumFixed = tNumEntries - tSum;
            LocalOrdinalVector tNodes("inactive nodes", tNumFixed);

            if (tNumFixed > 0)
            {
                // create a list of nodes with zero mask values
                OrdinalT tOffset(0);
                Kokkos::parallel_scan (Kokkos::RangePolicy<OrdinalT>(0,tNumEntries),
                KOKKOS_LAMBDA (const OrdinalT& iOrdinal, OrdinalT& aUpdate, const bool& tIsFinal)
                {
                    const OrdinalT tVal = tNodeMask(iOrdinal);
                    if( tIsFinal && !tVal ) { tNodes(aUpdate) = iOrdinal; }
                    aUpdate += (1-tVal);
                }, tOffset);
            }

            return tNodes;
        }

    };

    template <int mSpaceDim>
    class BrickMask : public Plato::Mask<mSpaceDim>
    {
      private:
        using Plato::Mask<mSpaceDim>::mCellMask;

        Plato::BoxLimits<mSpaceDim> mLimits;

      public:
        /******************************************************************************//**
         * \brief Constructor for Plato::Mask
         * \param [in] aMesh Omega_h mesh
         * \param [in] aInputParams Mask definition
        **********************************************************************************/
        BrickMask(
                  Omega_h::Mesh          & aMesh,
            const Teuchos::ParameterList & aInputParams
        ) :
            Plato::Mask<mSpaceDim>(aMesh),
            mLimits(aInputParams)
        {
            initialize(aMesh, aInputParams);
        }

        void initialize(
                  Omega_h::Mesh          & aMesh,
            const Teuchos::ParameterList & aInputParams
        )
        {
            auto tCellCenters = this->getCellCenters(aMesh);

            auto tCellMask = mCellMask;
            auto tLimits = mLimits;
            auto tNumCells = tCellCenters.extent(0);
            Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
            {
                tCellMask(aCellOrdinal) = 1;

                for (Plato::OrdinalType tDim=0; tDim<mSpaceDim; tDim++)
                {
                    auto tVal = tCellCenters(aCellOrdinal, tDim);
                    if (tVal > tLimits.mMaximum[tDim]) tCellMask(aCellOrdinal) = 0;
                    if (tVal < tLimits.mMinimum[tDim]) tCellMask(aCellOrdinal) = 0;
                }
            }, "cell mask");

            this->computeNodeMask(aMesh);
        }
    };

    template <int mSpaceDim>
    class ConstructiveMask : public Plato::Mask<mSpaceDim>
    {
      private:
        using Plato::Mask<mSpaceDim>::mCellMask;

        Plato::BoxLimits<mSpaceDim> mLimits;

        std::vector<std::shared_ptr<Plato::ConstructivePrimitive>> mPrimitives;

      public:
        /******************************************************************************//**
         * \brief Constructor for Plato::Mask
         * \param [in] aMesh Omega_h mesh
         * \param [in] aInputParams Mask definition
        **********************************************************************************/
        ConstructiveMask(
                  Omega_h::Mesh          & aMesh,
            const Teuchos::ParameterList & aInputParams
        ) :
            Plato::Mask<mSpaceDim>(aMesh),
            mLimits(aInputParams)
        {
            initialize(aMesh, aInputParams);
        }

        void initialize(
                  Omega_h::Mesh          & aMesh,
            const Teuchos::ParameterList & aInputParams
        )
        {

            auto tPrimitivesParams = aInputParams.sublist("Primitives");
            for(auto tIndex = tPrimitivesParams.begin(); tIndex != tPrimitivesParams.end(); ++tIndex)
            {
                const auto & tEntry  = tPrimitivesParams.entry(tIndex);
                const auto & tMyName = tPrimitivesParams.name(tIndex);

                if (!tEntry.isList())
                {
                    THROWERR("Parameter in Primitives list not valid.  Expect lists only.");
                }   

                Teuchos::ParameterList& tPrimitiveParams = tPrimitivesParams.sublist(tMyName);

                if (!tPrimitiveParams.isType<std::string>("Type")) {
                    THROWERR("Primitive definition is missing required parameter 'Type'");
                }

                auto tType = tPrimitiveParams.get<std::string>("Type");
                if ( tType == "Brick" )
                {
                    mPrimitives.push_back(std::make_shared<Plato::BrickPrimitive>(tPrimitiveParams));
                }
                else
                {
                    THROWERR("Unknown Primitive type requested");
                }

            }

            auto tCellCenters = this->getCellCenters(aMesh);

            auto tCellMask = mCellMask;
            for (auto& tPrimitive : mPrimitives)
            {
                tPrimitive->apply(tCellMask, tCellCenters);
            }

            this->computeNodeMask(aMesh);
        }
    };
    template <int mSpaceDim>
    class MaskFactory
    {
      public:
        std::shared_ptr<Plato::Mask<mSpaceDim>>
        create(
                  Omega_h::Mesh          & aMesh,
            const Teuchos::ParameterList & aInputParams
        )
        {
            if (!aInputParams.isSublist("Mask"))
            {
                THROWERR("Required parameter list ('Mask') is missing.");
            }
            else
            {
                auto tMaskParams = aInputParams.sublist("Mask");
                if(!tMaskParams.isType<std::string>("Mask Type"))
                {
                    THROWERR("Parsing Mask: Required parameter ('Mask Type') is missing.");
                }
                else
                {
                    auto tMaskType = tMaskParams.get<std::string>("Mask Type");
                    if (tMaskType == "Brick")
                    {
                        return std::make_shared<Plato::BrickMask<mSpaceDim>>(aMesh, tMaskParams);
                    }
                    else
                    if (tMaskType == "Constructive")
                    {
                        return std::make_shared<Plato::ConstructiveMask<mSpaceDim>>(aMesh, tMaskParams);
                    }
                    else
                    {
                        THROWERR("Unknown 'Mask Type' requested");
                    }
                }
            }
        }
    };
} // namespace Plato
