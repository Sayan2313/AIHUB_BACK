from fastapi import status

class AppException(Exception):
    def __init__(self, message:str,status_code:int=status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

class ModelNotFound(AppException):
    def __init__(self):
        super().__init__(message="Model Not Found",status_code=status.HTTP_404_NOT_FOUND)
class ModelUnavailable(AppException):
    def __init__(self):
        super().__init__(message="Model Unavailable",status_code=status.HTTP_503_SERVICE_UNAVAILABLE)