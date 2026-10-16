#ifndef DFMARKER_API_H
#define DFMARKER_API_H

#include <stdint.h>

#if defined(_WIN32)
# if defined(DFMARKER_BUILD)
#  define DFMARKER_API __declspec(dllexport)
# else
#  define DFMARKER_API __declspec(dllimport)
# endif
#else
# define DFMARKER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#define DFMARKER_ENUM_BASE : int32_t
#else
#define DFMARKER_ENUM_BASE
#endif

/** @brief Marker 追踪系统对象句柄，由接口创建和释放 */
typedef void *DFMARKER_HANDLE;

/** @brief Marker 追踪系统运行状态，与每帧位姿质量状态独立 */
typedef enum DFMARKER_ENUM_BASE {
    DFMARKER_CREATED = 0, /**< 系统已创建，尚未启动 */
    DFMARKER_RUNNING = 1, /**< 系统运行中，可输入图像 */
    DFMARKER_STOPPED = 2, /**< 系统已停止，需要释放并重新创建后才能再次启动 */
    DFMARKER_ERROR = 3    /**< 系统发生处理异常，需要释放并重新创建 */
} dfmarker_status_t;

/** @brief 输入图像的像素格式 */
typedef enum DFMARKER_ENUM_BASE {
    DFMARKER_MONO8 = 0, /**< 单通道灰度图，每个像素 1 字节 */
    DFMARKER_RGB8 = 1,  /**< RGB 彩色图，每个像素 3 字节 */
    DFMARKER_BGR8 = 2   /**< BGR 彩色图，每个像素 3 字节 */
} dfmarker_image_format_t;

/** @brief 每帧位姿质量状态，LOST 和 WARN 不表示图像输入接口调用失败 */
typedef enum DFMARKER_ENUM_BASE {
    DFMARKER_LOST = 0,   /**< 未检测到可用于定位的已标定 Tag，无有效位姿 */
    DFMARKER_WARN = 1,   /**< 检测到已标定 Tag，但求解或质量检查未通过，无有效位姿 */
    DFMARKER_SINGLE = 2, /**< 单面 Tag 定位，深度与姿态约束较弱 */
    DFMARKER_MULTI = 3   /**< 多面 Tag 角点联合定位 */
} dfmarker_quality_t;
#undef DFMARKER_ENUM_BASE

/**
 * @struct dfmarker_image_t
 * @brief 输入图像信息
 * @remarks 图像数据由调用方持有，在 dfmarker_feed_image 返回前必须保持可读且不变，
 *          返回后可释放或修改。图像分辨率必须与配置中的相机标定一致。
 */
typedef struct {
    uint64_t timestamp_ns;        /**< 时间戳，单位：纳秒；同一句柄内严格递增，首帧允许为 0 */
    uint32_t width;                /**< 图像宽度，单位：像素 */
    uint32_t height;               /**< 图像高度，单位：像素 */
    uint64_t stride_bytes;         /**< 每行数据的跨度，单位：字节，允许包含行填充 */
    const uint8_t *data;           /**< 图像数据指针 */
    uint64_t data_size_bytes;      /**< 图像缓冲区大小，单位：字节 */
    dfmarker_image_format_t format; /**< 图像像素格式 */
} dfmarker_image_t;

/**
 * @struct dfmarker_pose_t
 * @brief 三维位置与姿态，具体坐标系由所属结果字段说明
 */
typedef struct {
    double position[3];    /**< 位置，顺序：x,y,z，单位：米 */
    double orientation[4]; /**< Hamilton 四元数，顺序：x,y,z,w */
} dfmarker_pose_t;

/**
 * @struct dfmarker_detection_t
 * @brief 单个 Tag 的检测结果
 */
typedef struct {
    int32_t tag_id;        /**< Tag ID */
    int32_t has_corners;   /**< 角点是否有效；重复 ID 导致歧义时为 0，角点数组清零 */
    double corners_px[8];  /**< 图像角点，顺序：x0,y0,...,x3,y3，单位：像素，沿用 ArUco 角点顺序 */
} dfmarker_detection_t;

/**
 * @struct dfmarker_tag_error_t
 * @brief 单个 Tag 的重投影误差信息
 */
typedef struct {
    int32_t tag_id;                /**< Tag ID */
    int32_t has_error;             /**< 重投影误差是否有效，1：有效；0：无效 */
    double reprojection_error_px;  /**< 重投影误差，单位：像素；has_error 为 0 时清零 */
} dfmarker_tag_error_t;

/**
 * @struct dfmarker_result_t
 * @brief Marker 追踪结果，包含位姿、质量状态和 Tag 检测信息
 * @remarks 结果及其数组、reason 指针仅在本次回调期间有效，需要保留时须复制标量及指针指向的数据。
 *          数组数量为 0 时对应指针为 NULL，Tag ID 按升序排列。
 *          has_pose 同时表示三个 pose 字段是否有效，无有效位姿时三个字段均清零。
 *          相对轨迹以首个接受的参考 Tag 位姿为原点，丢失或拒绝会断轨，恢复后仍保留该原点。
 */
typedef struct {
    uint64_t timestamp_ns;                     /**< 输入图像的原始时间戳，单位：纳秒 */
    uint64_t frame_index;                      /**< 有效输入帧编号，从 0 开始，包含丢失和拒绝帧 */
    uint64_t segment;                          /**< 轨迹分段编号，尚无接受位姿时为 0，首次接受后为 1 */
    dfmarker_quality_t quality;                /**< 当前帧位姿质量状态 */
    const char *reason;                        /**< 质量状态的原因说明，UTF-8 编码 */
    int32_t has_pose;                          /**< 三个位姿字段是否有效，1：有效；0：无效 */
    dfmarker_pose_t camera_pose;               /**< 参考 Tag 到相机光学系的位姿，光学系 +X 向右、+Y 向下、+Z 向前 */
    dfmarker_pose_t relative_pose;             /**< 当前参考 Tag 到首个接受的参考 Tag 坐标系的原始相对位姿 */
    dfmarker_pose_t smooth_relative_pose;      /**< 平滑相对位姿，坐标系同 relative_pose，仅用于显示 */
    int32_t has_reprojection_error;            /**< 整体重投影误差是否有效，1：有效；0：无效 */
    double reprojection_error_px;             /**< 整体重投影误差，单位：像素；无效时清零 */
    const dfmarker_detection_t *detections;    /**< Tag 检测结果数组 */
    uint32_t detection_count;                  /**< Tag 检测结果数量 */
    const int32_t *visible_ids;                /**< 当前帧检测到的 Tag ID 数组 */
    uint32_t visible_count;                    /**< 检测到的 Tag ID 数量 */
    const int32_t *used_ids;                   /**< 定位结果使用的 Tag ID 数组 */
    uint32_t used_count;                       /**< 定位结果使用的 Tag ID 数量 */
    const int32_t *rejected_ids;               /**< 求解过程中报告的被拒绝 Tag ID 数组 */
    uint32_t rejected_count;                   /**< 被拒绝 Tag ID 数量 */
    const dfmarker_tag_error_t *tag_errors;    /**< 逐 Tag 重投影误差数组 */
    uint32_t tag_error_count;                  /**< 逐 Tag 重投影误差记录数量 */
} dfmarker_result_t;

/**
 * @brief Marker 追踪结果回调函数
 * @param result 当前帧追踪结果，仅在本次回调期间有效，需要保留时由调用方复制
 * @param user_data 注册回调时传入的用户数据
 * @remarks 回调在 dfmarker_feed_image 的调用线程中同步执行，不得抛出异常。
 *          回调期间拒绝对同一句柄的接口调用，包括查询状态和释放句柄；不同句柄相互独立。
 */
typedef void (*dfmarker_result_callback_t)(const dfmarker_result_t *result, void *user_data);

/**
 * @brief 接口通用调用约定
 * @remarks 同一句柄的调用须由调用方串行执行，不能并发访问。
 *          句柄参数只能传入 NULL 或由 dfmarker_create 创建且尚未释放的有效句柄。
 *          接口以返回值报告成功或失败，不向调用方抛出 C++ 异常。
 *          非法输入不消费帧编号或时间戳，不推进追踪状态；意外处理异常使系统进入 DFMARKER_ERROR。
 */

/**
 * @brief 创建 Marker 追踪系统
 * @param config_path SDK JSON 配置文件路径及文件名
 * @param handle 输出系统对象句柄的地址，调用前 *handle 必须为 NULL
 * @return 1：创建成功；0：创建失败
 * @remarks 创建失败时输出句柄保持为 NULL，已有有效句柄不会被覆盖。
 *          rig 的相对路径以 config_path 所在目录为基准解析。
 *          创建或处理失败的诊断信息输出到标准错误流 stderr。
 */
DFMARKER_API int32_t dfmarker_create(const char *config_path, DFMARKER_HANDLE *handle);

/**
 * @brief 配置 Marker 追踪结果回调函数
 * @param handle 系统对象句柄
 * @param callback 结果回调函数，传入 NULL 表示注销回调
 * @param user_data 回调时传回的用户数据
 * @return 1：配置成功；0：配置失败
 */
DFMARKER_API int32_t dfmarker_register_result_callback(
    DFMARKER_HANDLE handle, dfmarker_result_callback_t callback, void *user_data);

/**
 * @brief 启动 Marker 追踪系统
 * @param handle 系统对象句柄
 * @return 1：启动成功；0：启动失败
 * @remarks 重复启动运行中的系统返回成功；已停止或发生处理异常的系统不能再次启动，须释放并重新创建。
 */
DFMARKER_API int32_t dfmarker_start(DFMARKER_HANDLE handle);

/**
 * @brief 输入图像信息并执行当前帧 Marker 追踪
 * @param handle 已启动的系统对象句柄
 * @param image 输入图像信息，数据在接口返回前必须保持可读且不变
 * @return 1：输入并处理成功；0：输入或处理失败
 * @remarks 注册回调后，每个有效输入帧处理完成时同步回调一次，包含 LOST 和 WARN 帧。
 *          无有效位姿不表示接口调用失败，应通过回调中的 quality 和 has_pose 判断定位结果。
 *          未注册回调时仍执行追踪并推进帧编号和时间戳。
 */
DFMARKER_API int32_t dfmarker_feed_image(DFMARKER_HANDLE handle, const dfmarker_image_t *image);

/**
 * @brief 获取 Marker 追踪系统运行状态
 * @param handle 系统对象句柄
 * @param status 输出系统运行状态，不是当前帧位姿质量状态
 * @return 1：获取成功；0：获取失败
 */
DFMARKER_API int32_t dfmarker_get_status(DFMARKER_HANDLE handle, dfmarker_status_t *status);

/**
 * @brief 停止 Marker 追踪系统
 * @param handle 系统对象句柄
 * @return 1：停止成功；0：停止失败
 * @remarks 重复停止已停止的系统返回成功；发生处理异常的系统返回失败。
 *          停止后须释放并重新创建系统，重新创建也会建立新的相对轨迹原点。
 */
DFMARKER_API int32_t dfmarker_stop(DFMARKER_HANDLE handle);

/**
 * @brief 释放 Marker 追踪系统，并将调用方的句柄置空
 * @param handle 系统对象句柄的地址，不能为 NULL
 * @return 1：释放成功；0：释放失败
 * @remarks 对已经置空的句柄重复释放返回成功，回调期间不能释放当前句柄。
 */
DFMARKER_API int32_t dfmarker_destroy(DFMARKER_HANDLE *handle);

#ifdef __cplusplus
}
#endif
#endif
