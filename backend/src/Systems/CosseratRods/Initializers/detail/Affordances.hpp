#pragma once

#include <iostream>
#include <tuple>
#include <type_traits>

#include "Systems/CosseratRods/BlockInitializer.hpp"
#include "Utilities/CRTP.hpp"
#include "Utilities/NamedType.hpp"
#include "Utilities/TMPL.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace detail {

      template <typename Initializer>
      class EnableCopyInitialization
          : public CRTPHelper<Initializer, EnableCopyInitialization> {
       private:
        using CRTP = CRTPHelper<Initializer, EnableCopyInitialization>;
        using CRTP::self;

       public:
        template <typename... Args>
        decltype(auto) but_with(Args&&... args) const noexcept {
          // these args should replace those in straight cosserat rod
          // rest args should come from
          // make a copy, then replace the arguments.
          Initializer copy(self());
          EXPAND_PACK_LEFT_TO_RIGHT(
              (std::get<Args>(copy.data()) = std::forward<Args>(args)));
          return copy;
        }
      };

      template <typename Initializer>
      class EnableOptionalParameterInitialization
          : public CRTPHelper<Initializer,
                              EnableOptionalParameterInitialization> {
       private:
        using CRTP =
            CRTPHelper<Initializer, EnableOptionalParameterInitialization>;
        using CRTP::self;

        template <typename... Tags>
        auto get_subset_of(tmpl::list<Tags...>) const /*noexcept*/ {
          // may not be noexcept, but whatever
          return std::make_tuple(
              std::get<Tags>(self().default_initializers())...);
        }

       protected:
        template <typename... ExtraInitializers>
        decltype(auto) get_optional_parameters(
            ExtraInitializers&&... extra_initializers) const /*noexcept*/ {
          using default_initializers =
              typename Initializer::DefaultInitializers;
          using Mapping = ::blocks::FormMap::from<::named_type::tag_type>::to<
              default_initializers>;
          // template <typename TagType>
          // using initializer_from = tmpl::lookup<Mapping, TagType>;

          using default_initializer_tags =
              tmpl::transform<default_initializers,
                              tmpl::bind<::named_type::tag_type, tmpl::_1>>;

          using extra_initializer_tags = tmpl::list<
              ::named_type::tag_type<std::decay_t<ExtraInitializers>>...>;
          using tag_difference = tmpl::list_difference<default_initializer_tags,
                                                       extra_initializer_tags>;
          // pass in extra, and then pass in tag_diff from defaults
          using default_initializers_needed = tmpl::transform<
              tag_difference,
              tmpl::bind<tmpl::lookup, tmpl::pin<Mapping>, tmpl::_1>>;
          // using DefaultInitializersNeededCache =
          //     tmpl::as_tuple<default_initializers_needed>;

          //          auto needed_cache = [this]() {
          //            DefaultInitializersNeededCache needed_cache;
          //            cpp17::apply(, self().default_initializers())
          //                tmpl::for_each<default_initializers_needed>([&](auto
          //                di) {
          //                  using DefaultInitializer =
          //                  tmpl::type_from<decltype(di)>;
          //                  // force a copy here
          //                  std::get<DefaultInitializer>(needed_cache) =
          //                      std::get<DefaultInitializer>(
          //                          self().default_initializers());
          //                });
          //            return needed_cache;
          //          }();
          return std::tuple_cat(std::make_tuple(std::forward<ExtraInitializers>(
                                    extra_initializers)...),
                                get_subset_of(default_initializers_needed{}));
        }
      };

      template <typename Initializer>
      class InitializationInterface
          : public CRTPHelper<Initializer, InitializationInterface> {
       private:
        using CRTP = CRTPHelper<Initializer, InitializationInterface>;
        using CRTP::self;

       public:
        template <typename Plugin>
        decltype(auto) initialize() const noexcept {
          return cosserat_rod::initialize_cosserat_rod<Plugin>(std::tuple_cat(
              self().get_required_parameters(), self().default_initializers()));
        }

        // can be cosserat rod plugin or tagged plugin
        template <typename Plugin, typename TaggedInitializer,
                  typename... ExtraTaggedInitializers>
        decltype(auto) initialize(
            TaggedInitializer&& initializer,
            ExtraTaggedInitializers&&... extra_initializers) const
        /*noexcept*/ {
          // cannot automate this fully as as we repack it into functors that
          // can give out vectors or tensors
          return cosserat_rod::initialize_cosserat_rod<Plugin>(std::tuple_cat(
              // pass directly as references
              self().get_required_parameters(),
              // pass after manipulation
              self().get_optional_parameters(
                  std::forward<TaggedInitializer>(initializer),
                  std::forward<ExtraTaggedInitializers>(
                      extra_initializers)...)));
        }
      };

      template <typename Initializer, template <typename> class... Interfaces>
      struct DeriveAndEnableFriendshipWith : public Interfaces<Initializer>... {
        using Friends = tmpl::list<Interfaces<Initializer>...>;
      };

      template <typename Initializer>
      struct CosseratRodInitializerInterface
          : public DeriveAndEnableFriendshipWith<
                Initializer, detail::EnableCopyInitialization,
                detail::EnableOptionalParameterInitialization,
                detail::InitializationInterface> {};

    }  // namespace detail

  }  // namespace cosserat_rod

}  // namespace elastica
