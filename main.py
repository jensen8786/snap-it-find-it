import cv2
import pandas as pd
from io import BytesIO, StringIO
from PIL import Image
import math
import os.path
import image_search #this is to search similar image from extracted features
import object_detection_background_removal #this is using detectron
import logging
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackContext,
)

BOT_TOKEN = "replace_this_with_your_token"

# the database of ikea and hipvan products
ikea_data = pd.read_csv('data/ikea_products.csv', index_col=False)
hipvan_data = pd.read_csv('data/hipvan_products.csv', index_col=False)


# ## conversation bot
# https://github.com/python-telegram-bot/python-telegram-bot/blob/master/examples/conversationbot.py

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)

START, PHOTO, SELECT_OBJ = range(3)
    
            
def start(update: Update, _: CallbackContext) -> int:

    update.message.reply_text(
        'This bot can fetch you similar products from Ikea! '
        'Simply send me a picture to tell me what you are looking for. '
        'If the object is not detected in the first try, crop the image and resubmit the photo!\n\n  ',
    )

    return PHOTO


def photo(update: Update, context: CallbackContext) -> int:
    
    photo_file = update.message.photo[-1].get_file()
    photo_file.download('user_photo.jpg')
    img = cv2.cvtColor(cv2.imread("user_photo.jpg"), cv2.COLOR_BGR2RGB)
    update.message.reply_text(
            'It may take up to 10 seconds to process your image, please be patient...'
        )
    context.user_data['ori_img'] = img
    img_w_label, classes, output_img = object_detection_background_removal.detect_object(img)
    data = dict( zip( classes, output_img))
    context.user_data['data'] = data

    if len(classes) > 1:
        classes.append('Entire photo')
        col = 3
        row = math.ceil(len(classes) / col)
        
        #rearrange the list to list of list to show up better on screen
        #https://stackoverflow.com/questions/10124751/convert-a-flat-list-to-list-of-lists-in-python
        classes_rearrange = [classes[col*i : col*(i+1)] for i in range(row)]
        reply_keyboard = classes_rearrange
        markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True)
        update.message.reply_text(
            '👍 {} objects detected. Please select one object to search.'.format(len(classes)-1),
            reply_markup=markup,
        )

        img_crop_pil = Image.fromarray(img_w_label)
        byte_io = BytesIO()
        img_crop_pil.save(byte_io, format="JPEG")
        jpg_buffer = byte_io.getvalue()
        byte_io.close()
        update.message.reply_photo(jpg_buffer) 

        return SELECT_OBJ
    
    else:
        img_to_search = context.user_data['ori_img']
        update.message.reply_text(
            'Searching for matching products...'
        )

        img_path = image_search.search_image(img_to_search, 20) 
        
        count = 0
        for i, path in enumerate(img_path):
            if os.path.isfile(path):
                try:
                    if path.split('/')[0] == 'ikea_image':
                        caption = ikea_data[ikea_data.id == path.split('/')[-1].strip('.jpg').split('_')[-1]].pipUrl.iloc[0]
                    elif path.split('/')[0] == 'hipvan_image':
                        main_URL = 'http://hipvan.com/'
                        caption = main_URL + hipvan_data[hipvan_data.id == int(path.split('/')[-1].strip('.jpg').split('_')[1])].relative_url.iloc[0]
                    update.message.reply_text('Top {} match ⤵️'.format(count+1))
                    update.message.reply_photo(open(path, 'rb'), caption=caption) # replay to bot
                    count += 1
                    if count == 5:
                        break
                except:
                    pass
                    
        return ConversationHandler.END

def select_obj(update: Update, context: CallbackContext) -> int:
    
    selected_class = update.message.text
    if selected_class in list(context.user_data['data'].keys()):
        update.message.reply_text(
            'You have selected {}, searching for matching products...'.format(selected_class)
        )

        data = context.user_data['data']
        img_to_search = data[selected_class]
        img_path = image_search.search_image(img_to_search, 20) 
        
        count = 0
        for i, path in enumerate(img_path):
            if os.path.isfile(path):
                try:
                    if path.split('/')[0] == 'ikea_image':
                        caption = ikea_data[ikea_data.id == path.split('/')[-1].strip('.jpg').split('_')[-1]].pipUrl.iloc[0]
                    elif path.split('/')[0] == 'hipvan_image':
                        main_URL = 'http://hipvan.com/'
                        caption = main_URL + hipvan_data[hipvan_data.id == int(path.split('/')[-1].strip('.jpg').split('_')[1])].relative_url.iloc[0]
                    update.message.reply_text('Top {} match ⤵️'.format(count+1))
                    update.message.reply_photo(open(path, 'rb'), caption=caption) # replay to bot
                    count += 1
                    if count == 5:
                        break
                except:
                    pass
    
    elif selected_class == 'Entire photo':
        update.message.reply_text(
            'Searching for matching products using entire photo...'
        )

        img_to_search = context.user_data['ori_img']
        img_path = image_search.search_image(img_to_search, 20) 
        
        count = 0
        for i, path in enumerate(img_path):
            if os.path.isfile(path):
                try:
                    if path.split('/')[0] == 'ikea_image':
                        caption = ikea_data[ikea_data.id == path.split('/')[-1].strip('.jpg').split('_')[-1]].pipUrl.iloc[0]
                    elif path.split('/')[0] == 'hipvan_image':
                        main_URL = 'http://hipvan.com/'
                        caption = main_URL + hipvan_data[hipvan_data.id == int(path.split('/')[-1].strip('.jpg').split('_')[1])].relative_url.iloc[0]
                    update.message.reply_text('Top {} match ⤵️'.format(count+1))
                    update.message.reply_photo(open(path, 'rb'), caption=caption) # replay to bot
                    count += 1
                    if count == 5:
                        break
                except:
                    pass
 
    else:
        update.message.reply_text(
            'Your input is invalid, please upload a new photo to try again...'
        )

    return ConversationHandler.END


# for future enhancement
def cancel(update: Update, _: CallbackContext) -> int:
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    update.message.reply_text(
        'Bye! I hope we can talk again some day.', reply_markup=ReplyKeyboardRemove()
    )

    return ConversationHandler.END


def main() -> None:
    # Create the Updater and pass it your bot's token.
    updater = Updater(BOT_TOKEN)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # Add conversation handler with the states GENDER, PHOTO, LOCATION and BIO
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start),
                     MessageHandler(Filters.photo, photo),
                     ],
        states={
            START: [CommandHandler('start', start)], 
            PHOTO: [MessageHandler(Filters.photo, photo)],#, CommandHandler('skip', skip_photo)],
            SELECT_OBJ: [MessageHandler(Filters.text, select_obj), MessageHandler(Filters.photo, photo)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    dispatcher.add_handler(conv_handler)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


main()




