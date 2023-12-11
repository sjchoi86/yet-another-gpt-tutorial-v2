import sys
from gpt_helper import GPTchatClass

# Parse input text
input_text = sys.argv[1]

# Chat with GPT
GPT = GPTchatClass(
    gpt_model = "gpt-4",  # 'gpt-3.5-turbo' / 'gpt-4'
    role_msg  = "Your are a helpful assistant summarizing infromation and answering user queries.",
    key_path  = "../key/rilab_key.txt",
)

PRINT_USER_MSG   = False
PRINT_GPT_OUTPUT = False
RESET_CHAT       = False
RETURN_RESPONSE  = True
out = GPT.chat(
    user_msg         = input_text,
    PRINT_USER_MSG   = PRINT_USER_MSG,
    PRINT_GPT_OUTPUT = PRINT_GPT_OUTPUT,
    RESET_CHAT       = RESET_CHAT,
    RETURN_RESPONSE  = RETURN_RESPONSE,
)

# Output (do not modify this part)
print ("\nInput:")
print (input_text)
print ("\nOutput:")
print (out)
print ()
