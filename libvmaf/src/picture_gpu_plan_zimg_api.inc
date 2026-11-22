/*
 * Copied unchanged from zimg 3.0.6, src/zimg/api/zimg.cpp (https://github.com/sekrit-twc/zimg,
 * WTFPL licence): translate_cpu() to
 * translate_resize_filter() and import_graph_state_common() to
 * import_graph_params(), the C API's formats and parameters as zimg's graph
 * builder state. Included by picture_gpu_plan.cpp inside an unnamed namespace,
 * after search_enum_map() and search_itu_enum_map().
 */

zimg::CPUClass translate_cpu(zimg_cpu_type_e cpu)
{
	using zimg::CPUClass;

	static SM_CONSTEXPR_14 const zimg::static_map<zimg_cpu_type_e, CPUClass, 19> map{
		{ ZIMG_CPU_NONE,           CPUClass::NONE },
		{ ZIMG_CPU_AUTO,           CPUClass::AUTO },
		{ ZIMG_CPU_AUTO_64B,       CPUClass::AUTO_64B },
#if defined(ZIMG_X86)
		{ ZIMG_CPU_X86_MMX,        CPUClass::NONE },
		{ ZIMG_CPU_X86_SSE,        CPUClass::X86_SSE },
		{ ZIMG_CPU_X86_SSE2,       CPUClass::X86_SSE2 },
		{ ZIMG_CPU_X86_SSE3,       CPUClass::X86_SSE2 },
		{ ZIMG_CPU_X86_SSSE3,      CPUClass::X86_SSE2 },
		{ ZIMG_CPU_X86_SSE41,      CPUClass::X86_SSE2 },
		{ ZIMG_CPU_X86_SSE42,      CPUClass::X86_SSE2 },
		{ ZIMG_CPU_X86_AVX,        CPUClass::X86_AVX },
		{ ZIMG_CPU_X86_F16C,       CPUClass::X86_F16C },
		{ ZIMG_CPU_X86_AVX2,       CPUClass::X86_AVX2 },
		{ ZIMG_CPU_X86_AVX512F,    CPUClass::X86_AVX2 },
		{ ZIMG_CPU_X86_AVX512_SKX, CPUClass::X86_AVX512 },
		{ ZIMG_CPU_X86_AVX512_CLX, CPUClass::X86_AVX512_CLX },
		{ ZIMG_CPU_X86_AVX512_PMC, CPUClass::X86_AVX512 },
		{ ZIMG_CPU_X86_AVX512_SNC, CPUClass::X86_AVX512_CLX },
#elif defined(ZIMG_ARM)
		{ ZIMG_CPU_ARM_NEON_VFPv3, CPUClass::NONE },
		{ ZIMG_CPU_ARM_NEON_VFPv4, CPUClass::ARM_NEON },
#endif
	};
	return search_enum_map(map, cpu, "unrecognized cpu type");
}

zimg::PixelType translate_pixel_type(zimg_pixel_type_e pixel_type)
{
	using zimg::PixelType;

	static SM_CONSTEXPR_14 const zimg::static_map<zimg_pixel_type_e, zimg::PixelType, 4> map{
		{ ZIMG_PIXEL_BYTE,  PixelType::BYTE },
		{ ZIMG_PIXEL_WORD,  PixelType::WORD },
		{ ZIMG_PIXEL_HALF,  PixelType::HALF },
		{ ZIMG_PIXEL_FLOAT, PixelType::FLOAT },
	};
	return search_enum_map(map, pixel_type, "unrecognized pixel type");
}

bool translate_pixel_range(zimg_pixel_range_e range)
{
	static SM_CONSTEXPR_14 const zimg::static_map<zimg_pixel_range_e, bool, 2> map{
		{ ZIMG_RANGE_LIMITED, false },
		{ ZIMG_RANGE_FULL,    true },
	};
	return search_enum_map(map, range, "unrecognized pixel range");
}

zimg::graph::GraphBuilder::ColorFamily translate_color_family(zimg_color_family_e family)
{
	using zimg::graph::GraphBuilder;

	static SM_CONSTEXPR_14 const zimg::static_map<zimg_color_family_e, GraphBuilder::ColorFamily, 3> map{
		{ ZIMG_COLOR_GREY, GraphBuilder::ColorFamily::GREY },
		{ ZIMG_COLOR_RGB,  GraphBuilder::ColorFamily::RGB },
		{ ZIMG_COLOR_YUV,  GraphBuilder::ColorFamily::YUV },
	};
	return search_enum_map(map, family, "unrecognized color family");
}

zimg::graph::GraphBuilder::AlphaType translate_alpha(zimg_alpha_type_e alpha)
{
	using zimg::graph::GraphBuilder;

	static SM_CONSTEXPR_14 const zimg::static_map<zimg_alpha_type_e, GraphBuilder::AlphaType, 3> map{
		{ ZIMG_ALPHA_NONE,          GraphBuilder::AlphaType::NONE },
		{ ZIMG_ALPHA_STRAIGHT,      GraphBuilder::AlphaType::STRAIGHT },
		{ ZIMG_ALPHA_PREMULTIPLIED, GraphBuilder::AlphaType::PREMULTIPLIED },
	};
	return search_enum_map(map, alpha, "unrecognized alpha type");
}

zimg::graph::GraphBuilder::FieldParity translate_field_parity(zimg_field_parity_e field)
{
	using zimg::graph::GraphBuilder;

	static SM_CONSTEXPR_14 const zimg::static_map<zimg_field_parity_e, GraphBuilder::FieldParity, 3> map{
		{ ZIMG_FIELD_PROGRESSIVE, GraphBuilder::FieldParity::PROGRESSIVE },
		{ ZIMG_FIELD_TOP,         GraphBuilder::FieldParity::TOP },
		{ ZIMG_FIELD_BOTTOM,      GraphBuilder::FieldParity::BOTTOM },
	};
	return search_enum_map(map, field, "unrecognized field parity");
}

std::pair<zimg::graph::GraphBuilder::ChromaLocationW, zimg::graph::GraphBuilder::ChromaLocationH> translate_chroma_location(zimg_chroma_location_e chromaloc)
{
	typedef zimg::graph::GraphBuilder::ChromaLocationW ChromaLocationW;
	typedef zimg::graph::GraphBuilder::ChromaLocationH ChromaLocationH;

	// Workaround for std::pair assignment operator missing constexpr.
	struct chroma_pair {
		ChromaLocationW first;
		ChromaLocationH second;

		operator std::pair<ChromaLocationW, ChromaLocationH>() const { return{ first, second }; }
	};

	static SM_CONSTEXPR_14 const zimg::static_map<zimg_chroma_location_e, chroma_pair, 6> map{
		{ ZIMG_CHROMA_LEFT,        { ChromaLocationW::LEFT,   ChromaLocationH::CENTER } },
		{ ZIMG_CHROMA_CENTER,      { ChromaLocationW::CENTER, ChromaLocationH::CENTER } },
		{ ZIMG_CHROMA_TOP_LEFT,    { ChromaLocationW::LEFT,   ChromaLocationH::TOP } },
		{ ZIMG_CHROMA_TOP,         { ChromaLocationW::CENTER, ChromaLocationH::TOP } },
		{ ZIMG_CHROMA_BOTTOM_LEFT, { ChromaLocationW::LEFT,   ChromaLocationH::BOTTOM } },
		{ ZIMG_CHROMA_BOTTOM,      { ChromaLocationW::CENTER, ChromaLocationH::BOTTOM } },
	};
	return search_enum_map(map, chromaloc, "unrecognized chroma location");
}

zimg::colorspace::MatrixCoefficients translate_matrix(zimg_matrix_coefficients_e matrix)
{
	using zimg::colorspace::MatrixCoefficients;

	static SM_CONSTEXPR_14 const zimg::static_map<zimg_matrix_coefficients_e, zimg::colorspace::MatrixCoefficients, 13> map{
		{ ZIMG_MATRIX_RGB,                      MatrixCoefficients::RGB },
		{ ZIMG_MATRIX_BT709,                    MatrixCoefficients::REC_709 },
		{ ZIMG_MATRIX_UNSPECIFIED,              MatrixCoefficients::UNSPECIFIED },
		{ ZIMG_MATRIX_FCC,                      MatrixCoefficients::FCC },
		{ ZIMG_MATRIX_BT470_BG,                 MatrixCoefficients::REC_601 },
		{ ZIMG_MATRIX_ST170_M,                  MatrixCoefficients::REC_601 },
		{ ZIMG_MATRIX_ST240_M,                  MatrixCoefficients::SMPTE_240M },
		{ ZIMG_MATRIX_YCGCO,                    MatrixCoefficients::YCGCO },
		{ ZIMG_MATRIX_BT2020_NCL,               MatrixCoefficients::REC_2020_NCL },
		{ ZIMG_MATRIX_BT2020_CL,                MatrixCoefficients::REC_2020_CL },
		{ ZIMG_MATRIX_CHROMATICITY_DERIVED_NCL, MatrixCoefficients::CHROMATICITY_DERIVED_NCL },
		{ ZIMG_MATRIX_CHROMATICITY_DERIVED_CL,  MatrixCoefficients::CHROMATICITY_DERIVED_CL },
		{ ZIMG_MATRIX_ICTCP,                    MatrixCoefficients::REC_2100_ICTCP },
	};
	return search_itu_enum_map(map, matrix, "unrecognized matrix coefficients");
}

zimg::colorspace::TransferCharacteristics translate_transfer(zimg_transfer_characteristics_e transfer)
{
	using zimg::colorspace::TransferCharacteristics;

	static SM_CONSTEXPR_14 const zimg::static_map<zimg_transfer_characteristics_e, TransferCharacteristics, 16> map{
		{ ZIMG_TRANSFER_BT709,         TransferCharacteristics::REC_709 },
		{ ZIMG_TRANSFER_UNSPECIFIED,   TransferCharacteristics::UNSPECIFIED },
		{ ZIMG_TRANSFER_ST240_M,       TransferCharacteristics::SMPTE_240M },
		{ ZIMG_TRANSFER_BT601,         TransferCharacteristics::REC_709 },
		{ ZIMG_TRANSFER_BT470_M,       TransferCharacteristics::REC_470_M },
		{ ZIMG_TRANSFER_BT470_BG,      TransferCharacteristics::REC_470_BG },
		{ ZIMG_TRANSFER_IEC_61966_2_4, TransferCharacteristics::XVYCC },
		{ ZIMG_TRANSFER_IEC_61966_2_1, TransferCharacteristics::SRGB },
		{ ZIMG_TRANSFER_BT2020_10,     TransferCharacteristics::REC_709 },
		{ ZIMG_TRANSFER_BT2020_12,     TransferCharacteristics::REC_709 },
		{ ZIMG_TRANSFER_LINEAR,        TransferCharacteristics::LINEAR },
		{ ZIMG_TRANSFER_LOG_100,       TransferCharacteristics::LOG_100 },
		{ ZIMG_TRANSFER_LOG_316,       TransferCharacteristics::LOG_316 },
		{ ZIMG_TRANSFER_ST2084,        TransferCharacteristics::ST_2084 },
		{ ZIMG_TRANSFER_ST428,         TransferCharacteristics::ST_428 },
		{ ZIMG_TRANSFER_ARIB_B67,      TransferCharacteristics::ARIB_B67 },
	};
	return search_itu_enum_map(map, transfer, "unrecognized transfer characteristics");
}

zimg::colorspace::ColorPrimaries translate_primaries(zimg_color_primaries_e primaries)
{
	using zimg::colorspace::ColorPrimaries;

	static SM_CONSTEXPR_14 const zimg::static_map<zimg_color_primaries_e, ColorPrimaries, 12> map{
		{ ZIMG_PRIMARIES_BT470_M,     ColorPrimaries::REC_470_M },
		{ ZIMG_PRIMARIES_BT470_BG,    ColorPrimaries::REC_470_BG },
		{ ZIMG_PRIMARIES_BT709,       ColorPrimaries::REC_709 },
		{ ZIMG_PRIMARIES_UNSPECIFIED, ColorPrimaries::UNSPECIFIED },
		{ ZIMG_PRIMARIES_ST170_M,     ColorPrimaries::SMPTE_C },
		{ ZIMG_PRIMARIES_ST240_M,     ColorPrimaries::SMPTE_C },
		{ ZIMG_PRIMARIES_FILM,        ColorPrimaries::FILM },
		{ ZIMG_PRIMARIES_BT2020,      ColorPrimaries::REC_2020 },
		{ ZIMG_PRIMARIES_ST428,       ColorPrimaries::XYZ },
		{ ZIMG_PRIMARIES_ST431_2,     ColorPrimaries::DCI_P3 },
		{ ZIMG_PRIMARIES_ST432_1,     ColorPrimaries::DCI_P3_D65 },
		{ ZIMG_PRIMARIES_EBU3213_E,   ColorPrimaries::JEDEC_P22 },
	};
	return search_itu_enum_map(map, primaries, "unrecognized color primaries");
}

zimg::depth::DitherType translate_dither(zimg_dither_type_e dither)
{
	using zimg::depth::DitherType;

	static SM_CONSTEXPR_14 const zimg::static_map<zimg_dither_type_e, DitherType, 4> map{
		{ ZIMG_DITHER_NONE,            DitherType::NONE },
		{ ZIMG_DITHER_ORDERED,         DitherType::ORDERED },
		{ ZIMG_DITHER_RANDOM,          DitherType::RANDOM },
		{ ZIMG_DITHER_ERROR_DIFFUSION, DitherType::ERROR_DIFFUSION },
	};
	return search_enum_map(map, dither, "unrecognized dither type");
}

std::unique_ptr<zimg::resize::Filter> translate_resize_filter(zimg_resample_filter_e filter_type, double param_a, double param_b)
{
	if (filter_type == ZIMG_RESIZE_UNRESIZE)
		return nullptr;

	try {
		switch (filter_type) {
		case ZIMG_RESIZE_POINT:
			return ztd::make_unique<zimg::resize::PointFilter>();
		case ZIMG_RESIZE_BILINEAR:
			return ztd::make_unique<zimg::resize::BilinearFilter>();
		case ZIMG_RESIZE_BICUBIC:
			param_a = std::isnan(param_a) ? zimg::resize::BicubicFilter::DEFAULT_B : param_a;
			param_b = std::isnan(param_b) ? zimg::resize::BicubicFilter::DEFAULT_C : param_b;
			return ztd::make_unique<zimg::resize::BicubicFilter>(param_a, param_b);
		case ZIMG_RESIZE_SPLINE16:
			return ztd::make_unique<zimg::resize::Spline16Filter>();
		case ZIMG_RESIZE_SPLINE36:
			return ztd::make_unique<zimg::resize::Spline36Filter>();
		case ZIMG_RESIZE_SPLINE64:
			return ztd::make_unique<zimg::resize::Spline64Filter>();
		case ZIMG_RESIZE_LANCZOS:
			param_a = std::isnan(param_a) ? zimg::resize::LanczosFilter::DEFAULT_TAPS : std::max(param_a, 1.0);
			return ztd::make_unique<zimg::resize::LanczosFilter>(static_cast<unsigned>(param_a));
		default:
			zimg::error::throw_<zimg::error::EnumOutOfRange>("unrecognized resampling filter");
		}
	} catch (const std::bad_alloc &) {
		zimg::error::throw_<zimg::error::OutOfMemory>();
	}
}

void import_graph_state_common(const zimg_image_format &src, zimg::graph::GraphBuilder::state *out)
{
	if (src.version >= API_VERSION_2_0) {
		out->width = src.width;
		out->height = src.height;
		out->type = translate_pixel_type(src.pixel_type);
		out->subsample_w = src.subsample_w;
		out->subsample_h = src.subsample_h;

		// Handle colorspace constants separately.
		out->color = translate_color_family(src.color_family);

		out->depth = src.depth ? src.depth : zimg::pixel_depth(out->type);
		out->fullrange = translate_pixel_range(src.pixel_range);

		out->parity = translate_field_parity(src.field_parity);
		std::tie(out->chroma_location_w, out->chroma_location_h) = translate_chroma_location(src.chroma_location);
	}
	if (src.version >= API_VERSION_2_1) {
		out->active_left = std::isnan(src.active_region.left) ? 0 : src.active_region.left;
		out->active_top = std::isnan(src.active_region.top) ? 0 : src.active_region.top;
		out->active_width = std::isnan(src.active_region.width) ? src.width : src.active_region.width;
		out->active_height = std::isnan(src.active_region.height) ? src.height : src.active_region.height;
	} else {
		out->active_left = 0.0;
		out->active_top = 0.0;
		out->active_width = src.width;
		out->active_height = src.height;
	}
	if (src.version >= API_VERSION_2_4)
		out->alpha = translate_alpha(src.alpha);
}

std::pair<zimg::graph::GraphBuilder::state, zimg::graph::GraphBuilder::state> import_graph_state(const zimg_image_format &src, const zimg_image_format &dst)
{
	API_VERSION_ASSERT(src.version);
	API_VERSION_ASSERT(dst.version);
	zassert_d(src.version == dst.version, "image format versions do not match");

	zimg::graph::GraphBuilder::state src_state{};
	zimg::graph::GraphBuilder::state dst_state{};

	import_graph_state_common(src, &src_state);
	import_graph_state_common(dst, &dst_state);

	if (src.version >= API_VERSION_2_0) {
		// Accept unenumerated colorspaces if they form the basic no-op case.
		if (src.color_family == dst.color_family &&
		    src.matrix_coefficients == dst.matrix_coefficients &&
		    src.transfer_characteristics == dst.transfer_characteristics &&
		    src.color_primaries == dst.color_primaries)
		{
			src_state.colorspace = zimg::colorspace::ColorspaceDefinition{};
			dst_state.colorspace = zimg::colorspace::ColorspaceDefinition{};
		} else {
			src_state.colorspace.matrix = translate_matrix(src.matrix_coefficients);
			src_state.colorspace.transfer = translate_transfer(src.transfer_characteristics);
			src_state.colorspace.primaries = translate_primaries(src.color_primaries);

			dst_state.colorspace.matrix = translate_matrix(dst.matrix_coefficients);
			dst_state.colorspace.transfer = translate_transfer(dst.transfer_characteristics);
			dst_state.colorspace.primaries = translate_primaries(dst.color_primaries);
		}
	}

	return{ src_state, dst_state };
}

zimg::graph::GraphBuilder::params import_graph_params(const zimg_graph_builder_params &src, std::unique_ptr<zimg::resize::Filter> filters[2])
{
	API_VERSION_ASSERT(src.version);

	zimg::graph::GraphBuilder::params params{};

	if (src.version >= API_VERSION_2_0) {
		filters[0] = translate_resize_filter(src.resample_filter, src.filter_param_a, src.filter_param_b);
		filters[1] = translate_resize_filter(src.resample_filter_uv, src.filter_param_a_uv, src.filter_param_b_uv);

		params.filter = filters[0].get();
		params.filter_uv = filters[1].get();
		params.unresize = src.resample_filter == ZIMG_RESIZE_UNRESIZE;
		params.dither_type = translate_dither(src.dither_type);
		params.cpu = translate_cpu(src.cpu_type);
	}
	if (src.version >= API_VERSION_2_2) {
		params.peak_luminance = src.nominal_peak_luminance;
		params.approximate_gamma = !!src.allow_approximate_gamma;
	}

	return params;
}
