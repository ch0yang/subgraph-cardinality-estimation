set(GQL_BATCH_EXPORT_FILE_LIST
        GQLBatchExport.cpp
        FilterVertices.cpp
        FilterVertices.h)

add_executable(GQLBatchExport.out ${GQL_BATCH_EXPORT_FILE_LIST})
target_link_libraries(GQLBatchExport.out graph utility)
